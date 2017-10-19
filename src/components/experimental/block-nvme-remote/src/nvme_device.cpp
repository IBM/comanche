/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */
//#define FORMAT_ON_INIT

#include <common/cycles.h>
#include <common/dump_utils.h>
#include <common/rand.h>
#include <common/utils.h>
#include <city.h>
#include <mutex>
#include <condition_variable>

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <spdk/nvme.h>

#include "eal_init.h"
#include "nvme_device.h"
#include "self_tests.h"
#include "csem.h"


/**< per worker IO queue for MODE_QUEUED */
static struct
{
  std::mutex       g_lock;
  struct rte_ring* ring_list[RTE_MAX_LCORE];
  unsigned         ring_list_last;
  unsigned         launched;
}
qm_state;


/** 
 * MODE_QUEUED - IO processing thread
 * 
 */
static int lcore_entry_point(void * arg)
{
  unsigned current_core = rte_lcore_id();
  PLOG("io thread: %u entered.", current_core);
  assert(arg);
  Nvme_queue * queue = static_cast<Nvme_queue*>(arg);
  assert(queue);
  assert(queue->device());

  char tmp[32];
  snprintf(tmp,32,"io_thread_queue_%u",current_core);

  struct rte_ring * ring = rte_ring_create(tmp,
                                           Nvme_device::QUEUED_MODE_IOQ_DEPTH,
                                           SOCKET_ID_ANY,
                                           0);
  if(ring == nullptr)
    throw General_exception("rte_ring_created failed for IO thread (MODE_QUEUE)");

  {
    std::lock_guard<std::mutex> g(qm_state.g_lock);
    qm_state.ring_list[qm_state.ring_list_last] = ring;
    qm_state.ring_list_last++;
    qm_state.launched++;
  }
  
  /* work loop */
  queued_io_descriptor_t * desc;
  while(!queue->device()->exit_io_threads()) {

    /* process completion */
    queue->process_completions();
    
    /* deque next desc */
    if(rte_ring_sc_dequeue(ring, (void**)&desc) == -ENOENT) {
      cpu_relax();
      continue;
    }

    /* dequeued item */
    assert(desc);
    assert(desc->magic == 0x10101010);

    /* submit to hardware */
    queue->submit_async_op_internal(desc);
  }

  /* clean up ring */
  rte_ring_free(ring);
  PLOG("io thread: %u exited.", rte_lcore_id());
  return 0;
}



Nvme_device::Nvme_device(const char* device_id,
                         int mode,
                         cpu_set_t * io_threads) :
  _mode(mode),
  _exit_io_threads(false)
{ 
  _probed_device.ns = nullptr;
  _probed_device.ctrlr = nullptr;
                   
  assert(device_id);
  
  /*  initialize DPDK/SPDK and drivers */
  DPDK::eal_init(0);
  
  initialize(device_id);
  //  self_test("metadata");
  assert(mode > 0);
  
  if(_mode == MODE_QUEUED) {

    if(!io_threads)
      throw API_exception("Nvme_device constructor in queued mode needs mask");
    
    unsigned total_threads = CPU_COUNT(io_threads);
    PLOG("comanche mode: Queued Mode");
    //    if(num_queues == 0) throw API_exception("queued mode requires > 0 queues");
    size_t core = 0;
    for(unsigned i=0;i<total_threads;i++)  {
      while(!CPU_ISSET(core,io_threads)) core++;
      if(!rte_lcore_is_enabled(core)) {
        PWRN("core (%ld) specified is not enabled",core);
      }
      else {
        PLOG("launching IO thread: %ld",core);
        auto q = allocate_io_queue_pair(1); /* namespace id */
        rte_eal_remote_launch(lcore_entry_point, q, core);
      }      
      core++;
    }

    /* wait for launching to complete before we return */
    while(qm_state.launched < total_threads) {
      PLOG("waiting for launch completion (launched=%d)..", qm_state.launched);
      usleep(500000);
    }
  }
  else {
    PLOG("comanche mode: Direct Mode");
  }
}

Nvme_device::~Nvme_device()
{
  /* wait for IO threds */
  if(_mode == MODE_QUEUED) {
    _exit_io_threads = true;
    rte_eal_mp_wait_lcore();
  }
  
  /* detach from device */
  //  spdk_nvme_detach(_controller); 
}


/**
 * Initialize memory, probe, attach and initialize device
 *
 *
 */
void
Nvme_device::initialize(const char* device_id)
{
  PLOG("Looking for NVMe Controller (%s)...", device_id);

  strncpy(_probed_device.device_id, device_id, sizeof(_probed_device.device_id));
  
  /*
   * Start the SPDK NVMe enumeration process.  probe_cb will be called
   *  for each NVMe controller found, giving our application a choice on
   *  whether to attach to each controller.  attach_cb will then be
   *  called for each controller after the SPDK NVMe driver has completed
   *  initializing the controller we chose to attach.
   */
  if (spdk_nvme_probe(NULL /* transport for NVMe-oF */,
                      (void*)&_probed_device,
                      probe_cb, attach_cb, NULL) != 0) {
    throw new Device_exception("spdk_nvme_probe() failed\n");
  }
  
  PLOG("Probe complete (%p,%p)", _probed_device.ctrlr, _probed_device.ns);
  
  if(!_probed_device.ctrlr)
    throw new General_exception("NVMe device (%s) not found (check VFIO/UIO binding)", device_id);
  if(!_probed_device.ns)
    throw new General_exception("NVMe device (%s) invalid namespace info", device_id);

#ifdef FORMAT_ON_INIT
  format(6); /* format with lbaf=6 */
#endif
  
  /* output some device information */
  const struct spdk_nvme_ctrlr_data * caps = get_controller_caps();
  PINF("[ctlr-info] sqes: min(%ld) max(%ld)", 1UL << caps->sqes.min, 1UL << caps->sqes.max);
  PINF("[ctlr-info] cqes: min(%ld) max(%ld)", 1UL << caps->cqes.min, 1UL << caps->cqes.max);
  PINF("[ctlr-info] awun: %u", caps->awun + 1);
  PINF("[ctlr-info] awupf: %u", caps->awupf + 1);
  PINF("[ctlr-info] acwu: %u", caps->acwu);
  PINF("[ctlr-info] fused op: %s", caps->fuses > 0 ? "Y" : "N");
  PINF("[ctlr-info] metadata size: %u", get_metadata_size());
  PINF("[ctlr-info] max IO size: %u", get_max_io_xfer_size());
  PINF("[ns-info] extended LBA support: %s",
       get_ns_flags() & SPDK_NVME_NS_EXTENDED_LBA_SUPPORTED ? "Y" : "N");
  auto nsdata = ns_data();
  PINF("[ns-info] metadata transfer as extended LBA: %s", nsdata->mc.extended ? "Y" : "N");
  PINF("[ns-info] metadata transfer as separate pointer: %s", nsdata->mc.pointer ? "Y" : "N");
  PINF("[ns-info] nsze: %lu", nsdata->nsze);
  PINF("[ns-info] ncap: %lu", nsdata->ncap);
}

struct spdk_nvme_ns *
Nvme_device::get_namespace(uint32_t namespace_id)
{
  assert(_probed_device.ns);
  if (spdk_nvme_ns_get_id(_probed_device.ns) == namespace_id) 
    return _probed_device.ns;
  else
    throw API_exception("namespace (%d) not found, id is actually (%d)",
                        namespace_id, spdk_nvme_ns_get_id(_probed_device.ns));

  return NULL;
}

/**
 * Allocate IO queues for a given namespace on the attached device
 *
 * @param namespace_id Namespace identifier (default=1)
 */
Nvme_queue*
Nvme_device::allocate_io_queue_pair(uint32_t namespace_id)
{
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);
  
  struct spdk_nvme_ns * ns = get_namespace(namespace_id);

  // create new IO queue pair for namespace
  //
  struct spdk_nvme_io_qpair_opts opts;
  spdk_nvme_ctrlr_get_default_io_qpair_opts(_probed_device.ctrlr, &opts, sizeof(opts));
  opts.io_queue_requests = 4096;
  opts.qprio = SPDK_NVME_QPRIO_URGENT;

  struct spdk_nvme_qpair* qpair =
    spdk_nvme_ctrlr_alloc_io_qpair(_probed_device.ctrlr, &opts, sizeof(opts));
  
  if (qpair == NULL)
    throw API_exception("unable to allocate IO queues in SPDK.");

  PLOG("allocating queue [%p] (%lu) on namespace:%d block size=%d", qpair,
       _queues.size(), namespace_id, spdk_nvme_ns_get_sector_size(ns));

  auto qp = new Nvme_queue(this, _queues.size(), qpair);
  if(!qp)
    throw General_exception("unable to allocate NVME queues");
  
  _queues.push_back(qp);

#ifdef CONFIG_DEBUG_EXTRA
  uint32_t flags = spdk_nvme_ns_get_flags(ns->ns);

  PLOG("flags = 0x%x", flags);
  PLOG("queue supports WRITE ZEROs feature: %s",
       flags & SPDK_NVME_NS_WRITE_ZEROES_SUPPORTED ? "yes" : "no");
#endif

  return qp;
}

void*
Nvme_device::allocate_io_buffer(size_t num_bytes, bool zero_init, int numa_socket)
{
  size_t block_size = get_block_size();
  size_t rounded_up_size;

  if (num_bytes % block_size == 0)
    rounded_up_size = block_size;
  else
    rounded_up_size = ((num_bytes / block_size) + 1) * block_size;

  assert(num_bytes % block_size == 0);

  void * ptr;
  if (zero_init) {
    ptr = rte_zmalloc_socket(NULL, rounded_up_size, 64 /*alignment*/, numa_socket);
  }
  else {
    ptr = rte_malloc_socket(NULL, rounded_up_size, 64 /*alignment*/, numa_socket);
  }

  PLOG("allocated Nvme_buffer @ phys:%lx", rte_malloc_virt2phy(ptr));

  if (!ptr)
    throw new Constructor_exception("rte_zmalloc failed in Buffer constructor");
  
  return ptr;
}

void
Nvme_device::free_io_buffer(void * buffer)
{
  rte_free(buffer);
}

size_t
Nvme_device::get_block_size(uint32_t namespace_id)
{
  auto ns = get_namespace(namespace_id);
  return spdk_nvme_ns_get_sector_size(ns);
}

uint64_t
Nvme_device::get_size_in_blocks(uint32_t namespace_id)
{
  auto ns = get_namespace(namespace_id);
  return spdk_nvme_ns_get_num_sectors(ns);
}

uint64_t
Nvme_device::get_size_in_bytes(uint32_t namespace_id)
{
  auto ns = get_namespace(namespace_id);
  return spdk_nvme_ns_get_num_sectors(ns) *
         spdk_nvme_ns_get_sector_size(ns);
}

uint64_t
Nvme_device::get_max_squeue_depth(uint32_t namespace_id)
{
  const struct spdk_nvme_ctrlr_data * caps = get_controller_caps();
  return (1UL << caps->sqes.max);
}


void
Nvme_device::format(unsigned lbaf, uint32_t namespace_id)
{
  int rc;
  struct spdk_nvme_format fmt;
  
  assert(_probed_device.ctrlr);

  /* hard wired for the moment */
  switch(lbaf) {
  case 1:
  case 2:
  case 4:
  case 5:
  case 6:
    fmt.ms = 1; /* Intel SSD's only suppot extended LBA metadata */
    break;
  default:
    fmt.ms = 0;
  }
       
  PWRN("Low-level formatting device..");
  // format the device

  fmt.lbaf = lbaf;

  fmt.pi = 0;  // protection information
  fmt.pil = 0; // pi location
  fmt.ses = 0; // secure erase
  rc = spdk_nvme_ctrlr_format(_probed_device.ctrlr, namespace_id, &fmt);
  if (rc == 0)
    PINF("device format (lbaf=%u) OK.", lbaf);
  else
    PERR("device format failed rc=%d", rc);
}

void
Nvme_device::self_test(std::string testname, uint32_t namespace_id)
{
  auto max_lba = get_size_in_blocks(namespace_id);
  auto ns = get_namespace(namespace_id);
  auto block_size = get_block_size(namespace_id);

  PINF("self test: (ns:%u) max_lba=%ld, block_size=%ld", namespace_id, max_lba,
       block_size);

  struct spdk_nvme_io_qpair_opts opts;
  spdk_nvme_ctrlr_get_default_io_qpair_opts(_probed_device.ctrlr, &opts, sizeof(opts));
  opts.qprio = SPDK_NVME_QPRIO_URGENT;
  
  struct spdk_nvme_qpair* qpair =
    spdk_nvme_ctrlr_alloc_io_qpair(_probed_device.ctrlr, &opts, sizeof(opts));
  
  void* buffer = rte_malloc("selftest", block_size, block_size);
  assert(buffer);

  /* initialize non-kernel random # generator */
  init_genrand64(rdtsc());

  if (testname == "metadata") {
    test_metadata(ns, qpair);
  }
  if (testname == "hotspot") {
    PINF("Running hotspot test...");
    for (size_t r = 1; r < (2 << 20); r *= 2) {
      PWRN("Self-testing hot spot writes (range=%ld)..", r);
      test_random_block(ns, qpair, buffer, r, 100000, true);
    }
  } else if (testname.substr(0, 11) == "randstripW:") {
    size_t block_count = std::stoi(testname.substr(11));
    PINF("block_count = %ld", block_count);
    test_rand_sequential_block(ns, qpair, buffer,
                               0,       // lba_min,
                               max_lba, // lba_max,
                               block_count, 1000000,
                               true); // write flag
  } else if (testname.substr(0, 11) == "randstripR:") {
    size_t block_count = std::stoi(testname.substr(11));
    PINF("block_count = %ld", block_count);
    test_rand_sequential_block(ns, qpair, buffer,
                               0,       // lba_min,
                               max_lba, // lba_max,
                               block_count, 1000000,
                               false); // write flag
  } else if (testname == "blkrandR") { // block random read (full scope)
    test_random_block(ns, qpair, buffer, max_lba, 10000000, false);
  } else if (testname.substr(0, 9) ==
             "blkrandR:") { // block random read (256MB scope)
    size_t sz = std::stoi(testname.substr(9));
    if (sz > max_lba) {
      PERR("exceeds bounds");
      return;
    }
    test_random_block(ns, qpair, buffer, MB(sz) / block_size, 1000000,
                      false);
  } else if (testname == "blkrandW") { // block random write (full scope)
    test_random_block(ns, qpair, buffer, max_lba, 10000000, true);
  } else if (testname.substr(0, 9) ==
             "blkrandW:") { // block random write (256MB scope)
    size_t sz = std::stoi(testname.substr(9));
    if (sz > max_lba) {
      PERR("exceeds bounds");
      return;
    }
    test_random_block(ns, qpair, buffer, MB(sz) / block_size, 1000000,
                      true);
  } else if (testname == "blkseqR") { // block sequential read (full scope)
    test_sequential_block(ns, qpair, buffer, 1, max_lba, 1000000, false);
  } else if (testname == "blkseqW") { // block sequential write (full scope)
    test_sequential_block(ns, qpair, buffer, 1, max_lba, 1000000, true);
  } else if (testname == "blkseqR2M") { // block sequential read (full scope)
    unsigned num_blocks = MB(2) / block_size;
    void* buffer = rte_malloc("selftest", MB(2), block_size);
    test_sequential_block(ns, qpair, buffer, num_blocks, max_lba, 100000,
                          false);
  } else if (testname == "blkseqW2M") { // block sequential write (full scope)
    unsigned num_blocks = MB(2) / block_size;
    void* buffer = rte_malloc("selftest", MB(2), block_size);
    test_sequential_block(ns, qpair, buffer, num_blocks, max_lba, 100000,
                          true);
  } else if (testname.substr(0, 6) == "skipW:") { // skip write
    size_t sz = std::stoi(testname.substr(6));
    PINF("Stride=%ld", sz);
    if (sz > max_lba) {
      PERR("exceeds bounds");
      return;
    }
    test_skip_block(ns, qpair, buffer, max_lba,
                    MB(sz) / block_size, // stride
                    48,                  // num strides
                    100000,              // iterations
                    true);
  } else if (testname == "condition") {
    PINF("Conditioning drive with random writes for 3 hrs ...");
    time_t now = time(NULL);
    while (time(NULL) < now + (60 * 60 * 3)) {
      test_random_block(ns, qpair, buffer, max_lba, 10000000, true);
    }
  } else if (testname == "conditionloop") {
    PINF("Conditioning drive with random writes ...");
    while (1)
      test_random_block(ns, qpair, buffer, max_lba, 10000000, true);
  } else if (testname == "paging") {
    PINF("Paging test ...");
    size_t seq_len_blocks = 4096; // 2MB of 512 blocks
    void* buffer1 =
      rte_malloc("selftest1", block_size * seq_len_blocks, block_size);
    assert(buffer);
    void* buffer2 =
      rte_malloc("selftest2", block_size * seq_len_blocks, block_size);
    assert(buffer);

    test_sequential_block_mixed(ns, qpair, buffer1, buffer2, seq_len_blocks,
                                max_lba, 10000);
  }

#if 0
  /* do write warm up */
  while(1) {
    if(destructive) {
      PWRN("Self-testing (destructive) device..");
      test_random_block(ns, qpair, buffer, max_lba, 1000000, true);
    }
  }
#endif

  rte_free(buffer);
}

void
Nvme_device::raw_read(unsigned long lba, uint32_t namespace_id)
{
  auto max_lba = get_size_in_blocks(namespace_id);
  auto ns = get_namespace(namespace_id);
  auto block_size = get_block_size(namespace_id);

  PINF("raw read: (ns:%u) max_lba=%ld, block_size=%ld", namespace_id, max_lba,
       block_size);
  
  struct spdk_nvme_io_qpair_opts opts;
  spdk_nvme_ctrlr_get_default_io_qpair_opts(_probed_device.ctrlr, &opts, sizeof(opts));
  opts.qprio = SPDK_NVME_QPRIO_URGENT;

  struct spdk_nvme_qpair* qpair =
    spdk_nvme_ctrlr_alloc_io_qpair(_probed_device.ctrlr, &opts, sizeof(opts));
  
  void* buffer = rte_malloc("selftest", block_size, block_size);
  assert(buffer);

  memset(buffer, 0xBE, block_size);

  int rc;
  int complete = 0;
  rc = spdk_nvme_ns_cmd_read(
    ns,
    qpair,
    buffer,
    lba, // uint64_t lba,
    1,   // uint32_t lba_count,
    [](void* arg, const struct spdk_nvme_cpl*) { *((int*)arg) = 1; },
    &complete,
    0); // uint32_t io_flags);
  if (rc != 0)
    throw General_exception("spdk_nvme_ns_cmd_read failed unexpectedly");

  while(!complete) {
    spdk_nvme_qpair_process_completions(qpair, 0 /* unlimited completions */);
  }

  hexdump(buffer, block_size);
}



status_t Nvme_device::queue_submit_sync_op(void *buffer, uint64_t lba, uint64_t lba_count, int op)
{
  Semaphore s;
  if(_mode != MODE_QUEUED)
    throw API_exception("API is not in queued mode!");

  /* synchronize on a semaphore */
  queue_submit_async_op(buffer, lba, lba_count, op, INT_MAX, 
                        [](int tag, void* arg)
                        {
                          Semaphore * s = (Semaphore *) arg;
                          s->post();
                        },
                        &s);
  s.wait();
  
  return S_OK;
}


void Nvme_device::queue_submit_async_op(void *buffer,
                                        uint64_t lba,
                                        uint64_t lba_count,
                                        int op,
                                        int tag,
                                        io_callback_t cb,
                                        void *arg)
{
  static __thread unsigned q_selector = 0;
  
  if(_mode != MODE_QUEUED)
    throw API_exception("API is not in queued mode!");

  /* pick a queue - for the moment round-robin */
  unsigned queueid = q_selector++ % qm_state.ring_list_last;

  struct rte_ring * ring = qm_state.ring_list[queueid];
  assert(ring);

  /* build descriptor */
  queued_io_descriptor_t * desc = new queued_io_descriptor_t;
  desc->buffer = buffer;
  desc->lba = lba;
  desc->lba_count = lba_count;
  desc->op = op;
  desc->tag = tag;
  desc->cb = cb;
  desc->arg = arg;
  desc->magic = 0x10101010;
  
  /* post onto FIFO ring (as multi-producer) */
  int rc;
  while((rc = rte_ring_mp_enqueue(ring,(void*) desc)) != 0) {
    //    PWRN("queued mode async op enqueue (tag=%d, queueid=%u) failed (%d): ring full", tag, queueid, rc);
    cpu_relax();
  }

}


const struct spdk_nvme_ctrlr_data *
Nvme_device::
get_controller_caps()
{
  auto caps = spdk_nvme_ctrlr_get_data(_probed_device.ctrlr);
  assert(caps);
  return caps;
}


uint32_t
Nvme_device::
get_max_io_xfer_size()
{
  return spdk_nvme_ns_get_max_io_xfer_size(_probed_device.ns);
}

uint32_t
Nvme_device::
get_metadata_size()
{
  return spdk_nvme_ns_get_md_size(_probed_device.ns);
}
uint32_t 
Nvme_device::
get_ns_flags()
{
  return spdk_nvme_ns_get_flags(_probed_device.ns);
}

uint64_t
Nvme_device::
get_serial_hash()
{
  const struct spdk_nvme_ctrlr_data * caps = get_controller_caps();
  return CityHash64(reinterpret_cast<const char*>(&caps->sn), 20);
}

