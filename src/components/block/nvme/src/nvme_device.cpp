/*
   Copyright [2017] [IBM Corporation]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#define DISABLE_SIGPROF_ON_IO_THREAD /*< prevents IO thread from being \
                                        profiled with gperftools */
//#define FORMAT_ON_INIT
//#define DISABLE_IO // TESTING only.

#include <city.h>
#include <common/cycles.h>
#include <common/dump_utils.h>
#include <common/exceptions.h>
#include <common/rand.h>
#include <common/utils.h>
#include <condition_variable>
#include <mutex>

#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <spdk/nvme.h>

#include "csem.h"
#include "eal_init.h"
#include "nvme_device.h"
#include "self_tests.h"

/**
 * Service a ring (SW queue), issuing to a single queue.  It is important
 * that a thread cannot get stuck in this function.
 *
 * @param ring
 * @param queue
 */
static void do_work(struct rte_ring* ring, Nvme_queue* queue) {
  IO_descriptor* desc = nullptr;
  Nvme_device* device = queue->device();

  /* active completion polling */
  queue->process_completions();

  /* deque next desc */
  if (unlikely(rte_ring_sc_dequeue(ring, (void**) &desc) != 0)) {
    return;
  }

  if (desc == nullptr) return;

  desc->queue = queue;

  assert(desc);

  /** check completion request */
  if (unlikely(desc->op == COMANCHE_OP_CHECK_COMPLETION)) {
    if (desc->tag == 0) {
      desc->status =
          queue->pending_remain() ? IO_STATUS_COMPLETE : IO_STATUS_PENDING;
    } else {
      desc->status = queue->check_completion(desc->tag) ? IO_STATUS_COMPLETE
                                                        : IO_STATUS_PENDING;
    }
  }
  /** submission request */
  else {
    assert(check_aligned(desc->buffer, 32));
    queue->push_pending_fifo(desc);
    assert(desc->buffer);
#ifndef DISABLE_IO
    queue->submit_async_op_internal(desc);
#else
    queue->remove_pending_fifo(desc);
    device->free_desc(desc);
#endif
  }
}

static struct rte_ring* register_ring(unsigned core, Nvme_device* device) {
  static unsigned index = 0;
  char tmp[32];
  snprintf(tmp, 32, "nvmeq_%u_%u", core, index++);

  struct rte_ring* ring = rte_ring_create(tmp, Nvme_device::IO_SW_QUEUE_DEPTH,
                                          numa_node_of_cpu(core), 0);
  if (ring == nullptr)
    throw General_exception("rte_ring_create failed for IO thread (name=%s)",
                            tmp);

  device->register_ring(core, ring);
  return ring;
}

/**
 * IO processing thread
 *
 */
static int lcore_entry_point(void* arg) {
  unsigned current_core = rte_lcore_id();
  PINF("NVMe queue IO thread (core %u) entered.", current_core);
  assert(arg);
  Nvme_queue* queue = static_cast<Nvme_queue*>(arg);
  assert(queue);
  assert(queue->device());

  /* mask SIGPROF */
#ifdef DISABLE_SIGPROF_ON_IO_THREAD
  {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGPROF);
    int s = pthread_sigmask(SIG_BLOCK, &set, NULL);
    assert(s == 0);
  }
#endif

  /* create and register work queue */
  struct rte_ring* ring = register_ring(current_core, queue->device());
  assert(ring);

  /* work loop */
  IO_descriptor* desc;
  while (!queue->device()->test_exit_io_threads()) {
    /* perform work cycle */
    do_work(ring, queue);

    /* process shared work */
    queue->execute_shared_work();
  }

  /* clean up ring */
  rte_ring_free(ring);
  PLOG("NVMe queue IO thread: %u exited.", rte_lcore_id());
  return 0;
}

struct poller_arg {
  struct rte_ring* msg_ring;
  Nvme_queue* queue;
  Nvme_device* device;
};

static void poller_cb(unsigned core, void* arg0) {
  struct poller_arg* arg = static_cast<struct poller_arg*>(arg0);
  do_work(arg->msg_ring, arg->queue);
}

static void* poller_cb_init(unsigned core, void* thisptr) {
  assert(thisptr);
  Nvme_device* this_ = static_cast<Nvme_device*>(thisptr);

  struct poller_arg* arg = new struct poller_arg;
  arg->queue = this_->allocate_io_queue_pair(
      this_->DEFAULT_NAMESPACE_ID); /* namespace id */
  assert(arg->queue);
  arg->msg_ring = register_ring(core, this_);
  assert(arg->msg_ring);
  arg->device = this_;
  return static_cast<void*>(arg);
}

static void poller_cb_cleanup(unsigned core, void* arg0) {
  assert(arg0);

  struct poller_arg* arg = static_cast<struct poller_arg*>(arg0);
  assert(arg->queue);
  assert(arg->msg_ring);

  rte_ring_free(arg->msg_ring);
  delete arg->queue;
  delete arg;
}

static std::string gen_desc_ring_name() {
  static unsigned tag = 0;
  std::string name = "descring-";
  name += std::to_string(tag++);
  return name;
}

Nvme_device::Nvme_device(const char* device_id, Core::Poller* poller)
    : _exit_io_threads(false),
      _desc_ring(gen_desc_ring_name(), DESC_RING_SIZE),
      _activate_io_threads(false),
      _pci_id(device_id) {
  _probed_device.ns = nullptr;
  _probed_device.ctrlr = nullptr;
  memset(_probed_device.device_id, 0, 1024);

  _exit_io_threads = true; /* mark this as non-IO thread mode */

  PLOG("Nvme_device (poller):%p", this);
  PLOG("Nvme_device: descriptor ring size %lu", DESC_RING_SIZE);

  assert(device_id);
  initialize(device_id);
  assert(_probed_device.ns);

  _volume_id = _probed_device.device_id;

  /* allocate descriptors */
  for (unsigned i = 0; i < DESC_RING_SIZE; i++) {
    _desc_ring.mp_enqueue(new IO_descriptor);
  }

  /* register with poller */
  if (!poller) throw API_exception("poller reference is null");

  poller->register_percore_task(poller_cb_init, poller_cb, poller_cb_cleanup,
                                this);
}

static std::atomic<int> desctag;

Nvme_device::Nvme_device(const char* device_id, cpu_mask_t& io_thread_mask)
    : _exit_io_threads(false),
      //    _desc_ring(std::to_string(get_serial_hash()), DESC_RING_SIZE), //  +
      //    std::to_string(device->get_serial_hash())
      _desc_ring(
          DESC_RING_SIZE),  //  + std::to_string(device->get_serial_hash())
      _activate_io_threads(true),
      _pci_id(device_id) {
  _probed_device.ns = nullptr;
  _probed_device.ctrlr = nullptr;

  PLOG("Nvme_device (IO threads):%p", this);
  PLOG("Nvme_device: descriptor ring size %lu", DESC_RING_SIZE);

  assert(device_id);
  initialize(device_id);
  assert(_probed_device.ns);
  //  self_test("metadata");

  _volume_id = _probed_device.device_id;

  /* allocate descriptors */
  for (unsigned i = 0; i < DESC_RING_SIZE; i++) {
    _desc_ring.mp_enqueue(new IO_descriptor);
  }

  /* configure work threads */
  unsigned total_threads = io_thread_mask.count();
  size_t core = 0;

  if (io_thread_mask.check_core(rte_get_master_lcore()))
    throw Constructor_exception("cannot use master core (%u)",
                                rte_get_master_lcore());

  for (unsigned i = 0; i < total_threads; i++) {
    while (!io_thread_mask.check_core(core)) {
      core++;
    }
    if (!rte_lcore_is_enabled(core)) {
      PWRN("Core (%ld) specified is not enabled", core);
    } else {
      PLOG("Launching IO thread: %ld", core);
      if (!_default_core) _default_core = core;

      auto q = allocate_io_queue_pair(DEFAULT_NAMESPACE_ID); /* namespace id */
      rte_eal_remote_launch(lcore_entry_point, q, core);

      _cores.push_back(core);
    }
    core++;
  }

  /* wait for launching to complete before we return */
  while (_qm_state.launched < total_threads) {
    usleep(5000);
  }
}

Nvme_device::~Nvme_device() {
  if (_activate_io_threads) {
    if (!_exit_io_threads) {
      /* wait for IO threds */
      _exit_io_threads = true;
    }
    for (auto& core : _cores) {
      rte_eal_wait_lcore(core);
    }
    PLOG("all IO threads joined.");
  }

  /* clear descriptor memory */
  while (!_desc_ring.empty()) {
    IO_descriptor* d = nullptr;
    _desc_ring.mc_dequeue(d);
    if (d) delete d;
  }
}

/**
 * Initialize memory, probe, attach and initialize device
 *
 *
 */
void Nvme_device::initialize(const char* device_id) {
  PLOG("Looking for NVMe Controller (%s)...", device_id);

  strncpy(_probed_device.device_id, device_id,
          sizeof(_probed_device.device_id));

  /*
   * Start the SPDK NVMe enumeration process.  probe_cb will be called
   *  for each NVMe controller found, giving our application a choice on
   *  whether to attach to each controller.  attach_cb will then be
   *  called for each controller after the SPDK NVMe driver has completed
   *  initializing the controller we chose to attach.
   */
  if (spdk_nvme_probe(NULL /* transport for NVMe-oF */, (void*) &_probed_device,
                      probe_cb, attach_cb, NULL) != 0) {
    throw new Device_exception("spdk_nvme_probe() failed\n");
  }

  PLOG("Probe complete (%p,%p) %s", _probed_device.ctrlr, _probed_device.ns,
       _probed_device.device_id);

  if (!_probed_device.ctrlr)
    throw new General_exception(
        "NVMe device (%s) not found (check VFIO/UIO binding)", device_id);
  if (!_probed_device.ns)
    throw new General_exception("NVMe device (%s) invalid namespace info",
                                device_id);

#ifdef FORMAT_ON_INIT
  format(3); /* format with lbaf=3 */
#endif

  /* output some device information */
  const struct spdk_nvme_ctrlr_data* caps = get_controller_caps();
  PINF("[ctlr-info] sqes: min(%ld) max(%ld)", 1UL << caps->sqes.min,
       1UL << caps->sqes.max);
  PINF("[ctlr-info] cqes: min(%ld) max(%ld)", 1UL << caps->cqes.min,
       1UL << caps->cqes.max);
  PINF("[ctlr-info] awun: %u", caps->awun + 1);
  PINF("[ctlr-info] awupf: %u", caps->awupf + 1);
  PINF("[ctlr-info] acwu: %u", caps->acwu);
  PINF("[ctlr-info] fused op: %s", caps->fuses > 0 ? "Y" : "N");
  PINF("[ctlr-info] metadata size: %u", get_metadata_size());
  PINF("[ctlr-info] max IO size: %u", get_max_io_xfer_size());
  PINF("[ns-info] extended LBA support: %s",
       get_ns_flags() & SPDK_NVME_NS_EXTENDED_LBA_SUPPORTED ? "Y" : "N");
  auto nsdata = ns_data();
  PINF("[ns-info] metadata transfer as extended LBA: %s",
       nsdata->mc.extended ? "Y" : "N");
  PINF("[ns-info] metadata transfer as separate pointer: %s",
       nsdata->mc.pointer ? "Y" : "N");
  PINF("[ns-info] nsze: %lu", nsdata->nsze);
  PINF("[ns-info] ncap: %lu", nsdata->ncap);
}

struct spdk_nvme_ns* Nvme_device::get_namespace(uint32_t namespace_id) {
  if (!_probed_device.ns)
    throw Logic_exception("probed device namespace ptr invalid");

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
Nvme_queue* Nvme_device::allocate_io_queue_pair(uint32_t namespace_id) {
  static std::mutex lock;
  std::lock_guard<std::mutex> guard(lock);

  struct spdk_nvme_ns* ns = get_namespace(namespace_id);

  // create new IO queue pair for namespace
  //
  struct spdk_nvme_io_qpair_opts opts;
  spdk_nvme_ctrlr_get_default_io_qpair_opts(_probed_device.ctrlr, &opts,
                                            sizeof(opts));
  opts.io_queue_requests = 4096;
  ;
  opts.qprio = SPDK_NVME_QPRIO_URGENT;

  struct spdk_nvme_qpair* qpair =
      spdk_nvme_ctrlr_alloc_io_qpair(_probed_device.ctrlr, &opts, sizeof(opts));

  if (qpair == NULL)
    throw API_exception("unable to allocate IO queues in SPDK.");

  PLOG("allocating queue [%p] (%lu) on namespace:%d block size=%d", qpair,
       _queues.size(), namespace_id, spdk_nvme_ns_get_sector_size(ns));

  auto qp = new Nvme_queue(this, _queues.size(), qpair);
  if (!qp) throw General_exception("unable to allocate NVME queues");

  _queues.push_back(qp);

#ifdef CONFIG_DEBUG_EXTRA
  uint32_t flags = spdk_nvme_ns_get_flags(ns->ns);

  PLOG("flags = 0x%x", flags);
  PLOG("queue supports WRITE ZEROs feature: %s",
       flags & SPDK_NVME_NS_WRITE_ZEROES_SUPPORTED ? "yes" : "no");
#endif

  return qp;
}

void* Nvme_device::allocate_io_buffer(size_t num_bytes, bool zero_init,
                                      int numa_socket) {
  size_t block_size = get_block_size();
  size_t rounded_up_size;

  if (num_bytes % block_size == 0)
    rounded_up_size = block_size;
  else
    rounded_up_size = ((num_bytes / block_size) + 1) * block_size;

  assert(num_bytes % block_size == 0);

  void* ptr;
  if (zero_init) {
    ptr = rte_zmalloc_socket(NULL, rounded_up_size, 64 /*alignment*/,
                             numa_socket);
  } else {
    ptr =
        rte_malloc_socket(NULL, rounded_up_size, 64 /*alignment*/, numa_socket);
  }

  PLOG("allocated Nvme_buffer @ phys:%lx", rte_malloc_virt2iova(ptr));

  if (!ptr)
    throw new Constructor_exception("rte_zmalloc failed in Buffer constructor");

  return ptr;
}

void Nvme_device::free_io_buffer(void* buffer) { rte_free(buffer); }

size_t Nvme_device::get_block_size(uint32_t namespace_id) {
  auto ns = get_namespace(namespace_id);
  return spdk_nvme_ns_get_sector_size(ns);
}

uint64_t Nvme_device::get_size_in_blocks(uint32_t namespace_id) {
  auto ns = get_namespace(namespace_id);
  return spdk_nvme_ns_get_num_sectors(ns);
}

uint64_t Nvme_device::get_size_in_bytes(uint32_t namespace_id) {
  auto ns = get_namespace(namespace_id);
  return spdk_nvme_ns_get_num_sectors(ns) * spdk_nvme_ns_get_sector_size(ns);
}

uint64_t Nvme_device::get_max_squeue_depth(uint32_t namespace_id) {
  const struct spdk_nvme_ctrlr_data* caps = get_controller_caps();
  return (1UL << caps->sqes.max);
}

const char* Nvme_device::get_device_id() {
  //  assert(_volume_id.empty() == false);
  return _volume_id.c_str();
}

const char* Nvme_device::get_pci_id() { return _pci_id.c_str(); }

void Nvme_device::format(unsigned lbaf, uint32_t namespace_id) {
  int rc;
  struct spdk_nvme_format fmt;

  assert(_probed_device.ctrlr);

  /* hard wired for the moment */
  switch (lbaf) {
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

  fmt.pi = 0;   // protection information
  fmt.pil = 0;  // pi location
  fmt.ses = 0;  // secure erase
  rc = spdk_nvme_ctrlr_format(_probed_device.ctrlr, namespace_id, &fmt);
  if (rc == 0)
    PINF("device format (lbaf=%u) OK.", lbaf);
  else
    PERR("device format failed rc=%d", rc);
}

void Nvme_device::self_test(std::string testname, uint32_t namespace_id) {
  auto max_lba = get_size_in_blocks(namespace_id);
  auto ns = get_namespace(namespace_id);
  auto block_size = get_block_size(namespace_id);

  PINF("self test: (ns:%u) max_lba=%ld, block_size=%ld", namespace_id, max_lba,
       block_size);

  struct spdk_nvme_io_qpair_opts opts;
  spdk_nvme_ctrlr_get_default_io_qpair_opts(_probed_device.ctrlr, &opts,
                                            sizeof(opts));
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
                               0,        // lba_min,
                               max_lba,  // lba_max,
                               block_count, 1000000,
                               true);  // write flag
  } else if (testname.substr(0, 11) == "randstripR:") {
    size_t block_count = std::stoi(testname.substr(11));
    PINF("block_count = %ld", block_count);
    test_rand_sequential_block(ns, qpair, buffer,
                               0,        // lba_min,
                               max_lba,  // lba_max,
                               block_count, 1000000,
                               false);  // write flag
  } else if (testname == "blkrandR") {  // block random read (full scope)
    test_random_block(ns, qpair, buffer, max_lba, 10000000, false);
  } else if (testname.substr(0, 9) ==
             "blkrandR:") {  // block random read (256MB scope)
    size_t sz = std::stoi(testname.substr(9));
    if (sz > max_lba) {
      PERR("exceeds bounds");
      return;
    }
    test_random_block(ns, qpair, buffer, MB(sz) / block_size, 1000000, false);
  } else if (testname == "blkrandW") {  // block random write (full scope)
    test_random_block(ns, qpair, buffer, max_lba, 10000000, true);
  } else if (testname.substr(0, 9) ==
             "blkrandW:") {  // block random write (256MB scope)
    size_t sz = std::stoi(testname.substr(9));
    if (sz > max_lba) {
      PERR("exceeds bounds");
      return;
    }
    test_random_block(ns, qpair, buffer, MB(sz) / block_size, 1000000, true);
  } else if (testname == "blkseqR") {  // block sequential read (full scope)
    test_sequential_block(ns, qpair, buffer, 1, max_lba, 1000000, false);
  } else if (testname == "blkseqW") {  // block sequential write (full scope)
    test_sequential_block(ns, qpair, buffer, 1, max_lba, 1000000, true);
  } else if (testname == "blkseqR2M") {  // block sequential read (full scope)
    unsigned num_blocks = MB(2) / block_size;
    void* buffer = rte_malloc("selftest", MB(2), block_size);
    test_sequential_block(ns, qpair, buffer, num_blocks, max_lba, 100000,
                          false);
  } else if (testname == "blkseqW2M") {  // block sequential write (full scope)
    unsigned num_blocks = MB(2) / block_size;
    void* buffer = rte_malloc("selftest", MB(2), block_size);
    test_sequential_block(ns, qpair, buffer, num_blocks, max_lba, 100000, true);
  } else if (testname.substr(0, 6) == "skipW:") {  // skip write
    size_t sz = std::stoi(testname.substr(6));
    PINF("Stride=%ld", sz);
    if (sz > max_lba) {
      PERR("exceeds bounds");
      return;
    }
    test_skip_block(ns, qpair, buffer, max_lba,
                    MB(sz) / block_size,  // stride
                    48,                   // num strides
                    100000,               // iterations
                    true);
  } else if (testname == "condition") {
    PINF("Conditioning drive with random writes for 3 hrs ...");
    time_t now = time(NULL);
    while (time(NULL) < now + (60 * 60 * 3)) {
      test_random_block(ns, qpair, buffer, max_lba, 10000000, true);
    }
  } else if (testname == "conditionloop") {
    PINF("Conditioning drive with random writes ...");
    while (1) test_random_block(ns, qpair, buffer, max_lba, 10000000, true);
  } else if (testname == "paging") {
    PINF("Paging test ...");
    size_t seq_len_blocks = 4096;  // 2MB of 512 blocks
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

void Nvme_device::raw_read(unsigned long lba, uint32_t namespace_id) {
  auto max_lba = get_size_in_blocks(namespace_id);
  auto ns = get_namespace(namespace_id);
  auto block_size = get_block_size(namespace_id);

  PINF("raw read: (ns:%u) max_lba=%ld, block_size=%ld", namespace_id, max_lba,
       block_size);

  struct spdk_nvme_io_qpair_opts opts;
  spdk_nvme_ctrlr_get_default_io_qpair_opts(_probed_device.ctrlr, &opts,
                                            sizeof(opts));
  opts.qprio = SPDK_NVME_QPRIO_URGENT;

  struct spdk_nvme_qpair* qpair =
      spdk_nvme_ctrlr_alloc_io_qpair(_probed_device.ctrlr, &opts, sizeof(opts));

  void* buffer = rte_malloc("selftest", block_size, block_size);
  assert(buffer);

  memset(buffer, 0xBE, block_size);

  int rc;
  int complete = 0;
  rc = spdk_nvme_ns_cmd_read(
      ns, qpair, buffer,
      lba,  // uint64_t lba,
      1,    // uint32_t lba_count,
      [](void* arg, const struct spdk_nvme_cpl*) { *((int*) arg) = 1; },
      &complete,
      0);  // uint32_t io_flags);
  if (rc != 0)
    throw General_exception("spdk_nvme_ns_cmd_read failed unexpectedly");

  while (!complete) {
    spdk_nvme_qpair_process_completions(qpair, 0 /* unlimited completions */);
  }

  hexdump(buffer, block_size);
}

void Nvme_device::queue_submit_async_op(void* buffer, uint64_t lba,
                                        uint64_t lba_count, int op,
                                        uint64_t tag, io_callback_t cb,
                                        void* arg0, void* arg1, int queue_id) {
  assert(buffer);

  if (queue_id == 0) queue_id = _default_core;

  struct rte_ring* ring = _qm_state.ring_list[queue_id];

  if (!ring)
    throw Logic_exception("invalid queue_id: qm_state[%d] not set up",
                          queue_id);

  if (option_DEBUG) {
    PINF("[-->FIFO] submit async op buffer:%p lba:%lu lbacount:%lu", buffer,
         lba, lba_count);
    assert(!((uint64_t) buffer & 0x1ULL));
  }

  /* build descriptor */
  IO_descriptor* desc = alloc_desc();
  assert(desc);

  desc->buffer = buffer;
  desc->lba = lba;
  desc->lba_count = lba_count;
  desc->op = op;
  desc->tag = tag;
  desc->cb = cb;
  desc->arg0 = arg0;
  desc->arg1 = arg1;

  wmb();

  /* post onto FIFO ring (as multi-producer) */
  int rc;
  unsigned retries = 0;
  while ((rc = rte_ring_mp_enqueue(ring, (void*) desc)) != 0) {
    //    PWRN("queued mode async op enqueue (tag=%d, queueid=%u) failed (%d):
    //    ring full", tag, queueid, rc);
    cpu_relax();
    if (retries++ > 100) {
      //      PWRN("push request onto FIFO queue jammed; too much pressure -
      //      backed off");
      usleep(100);
    }
  }
}

bool Nvme_device::check_completion(uint64_t gwid, int queue_id) {
  if (queue_id == 0) queue_id = _default_core;

  struct rte_ring* ring = _qm_state.ring_list[queue_id];
  if (!ring)
    throw API_exception(
        "invalid queue for check_completion; default may be invalid");

  /* build check completion request descriptor */
  IO_descriptor desc;  // = new IO_descriptor; // _desc_vector[_desc_vector_i];
  // while(!desc)
  //   desc_ring.mc_dequeue(desc);
  //  assert(desc);
  desc.op = COMANCHE_OP_CHECK_COMPLETION;
  desc.tag = gwid;
  desc.status = IO_STATUS_UNKNOWN;

  wmb();

  /* post onto FIFO ring (as multi-producer) */
  int rc;
  while ((rc = rte_ring_mp_enqueue(ring, (void*) &desc)) != 0) {
    cpu_relax();
  }

  /* wait for status update */
  cpu_time_t start = rdtsc();
  while (!desc.status) {
    if ((rdtsc() - start) > (2400 * 10000000UL)) {
      throw Logic_exception("check_completion timed out: (gwid=%lu)", gwid);
    }
    cpu_relax();
  }

  return (desc.status == IO_STATUS_COMPLETE);
}

bool Nvme_device::pending_remain() {
  return !check_completion(
      0);  // special gwid zero means check for all complete.
}

const struct spdk_nvme_ctrlr_data* Nvme_device::get_controller_caps() {
  auto caps = spdk_nvme_ctrlr_get_data(_probed_device.ctrlr);
  assert(caps);
  return caps;
}

uint32_t Nvme_device::get_max_io_xfer_size() {
  return spdk_nvme_ns_get_max_io_xfer_size(_probed_device.ns);
}

uint32_t Nvme_device::get_metadata_size() {
  return spdk_nvme_ns_get_md_size(_probed_device.ns);
}
uint32_t Nvme_device::get_ns_flags() {
  return spdk_nvme_ns_get_flags(_probed_device.ns);
}

uint64_t Nvme_device::get_serial_hash() {
  const struct spdk_nvme_ctrlr_data* caps = get_controller_caps();
  assert(caps);
  return CityHash64(reinterpret_cast<const char*>(&caps->sn),
                    SPDK_NVME_CTRLR_SN_LEN);
}

void Nvme_device::attach_work(unsigned queue_id,
                              std::function<void(void*)> work_function,
                              void* arg) {
  if (queue_id >= _queues.size())
    throw API_exception("%s: invalid queue", __PRETTY_FUNCTION__);

  Nvme_queue* q = _queues[queue_id];
  assert(q);
  q->attach_work(work_function, arg);
}
