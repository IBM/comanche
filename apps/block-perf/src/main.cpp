#include <api/block_itf.h>
#include <api/components.h>
#include <common/assert.h>
#include <common/exceptions.h>
#include <common/rand.h>
#include <common/spsc_bounded_queue.h>
#include <common/utils.h>
#include <core/dpdk.h>
#include <core/poller.h>
#include <core/task.h>
#include <core/xms.h>
#include <libpmem.h>
#include <libpmemobj.h>
#include <libpmempool.h>
#include <nupm/nd_utils.h>
#include <spdk/env.h>
#include <stdio.h>
#include <boost/program_options.hpp>
#include <string>
#include <vector>

using namespace std;

enum {
  WORKLOAD_RAND_READ = 1,
  WORKLOAD_RAND_WRITE = 2,
  WORKLOAD_RW = 5,
};

struct {
  unsigned n_client_threads;
  unsigned n_io_threads = 1;
  unsigned chunk_size_in_block = 8;
  int workload = WORKLOAD_RAND_READ;
} Options;

#define START_IO_CORE 2
#define START_CLIENT_CORE 6

static int check_pool(const char* path) {
  PMEMpoolcheck* ppc;
  struct pmempool_check_status* status;

  struct pmempool_check_args args;
  args.path = path;
  args.backup_path = NULL;
  args.pool_type = PMEMPOOL_POOL_TYPE_DETECT;
  args.flags = PMEMPOOL_CHECK_FORMAT_STR | PMEMPOOL_CHECK_REPAIR |
    PMEMPOOL_CHECK_VERBOSE;

  if ((ppc = pmempool_check_init(&args, sizeof(args))) == NULL) {
    perror("pmempool_check_init");
    return -1;
  }

  /* perform check and repair, answer 'yes' for each question */
  while ((status = pmempool_check(ppc)) != NULL) {
    switch (status->type) {
    case PMEMPOOL_CHECK_MSG_TYPE_ERROR:
    case PMEMPOOL_CHECK_MSG_TYPE_INFO:
      break;
    case PMEMPOOL_CHECK_MSG_TYPE_QUESTION:
      printf("%s\n", status->str.msg);
      status->str.answer = "yes";
      break;
    default:
      pmempool_check_end(ppc);
      return 1;
    }
  }

  /* finalize the check and get the result */
  int ret = pmempool_check_end(ppc);
  switch (ret) {
  case PMEMPOOL_CHECK_RESULT_CONSISTENT:
  case PMEMPOOL_CHECK_RESULT_REPAIRED:
    return 0;
  }

  return 1;
}

class Pmem_allocator
{
public:
  Pmem_allocator(const std::string path) noexcept(false) {
    int is_pmem;
    
    if((_base = pmem_map_file(path.c_str(),
                              0, /* size 0 because this is devdax */
                              PMEM_FILE_CREATE,
                              666,
                              &_len, &is_pmem)) == nullptr) {
        throw General_exception("pmem_map_file failed unexpectedly");
    }

    if(_len == 0)
      throw General_exception("pmem_map_file failed unexpectedly; len == 0");

    PLOG("Pmem_allocator using path (%s) mapped %p @ len %lu", path.c_str(), _base, _len);

    //    memset(_base, 0, GB(1)); //_len);
    
    if(spdk_mem_register(_base, _len))
      throw General_exception("spdk_mem_register failed");

    _phys_base = spdk_vtophys(_base);
  }

  ~Pmem_allocator() {
    pmem_unmap(_base, _len);
  }

  void * allocate(size_t len, addr_t& out_phys) {
    len = round_up(len, MB(2));
    if(_offset + len > _len)
      throw General_exception("out of persistent memory");
    void * rp = (void*)(((unsigned long long)_base) + _offset);
    out_phys = _phys_base + _offset;
    _offset += len;
    return rp;
  }
  
private:
  void * _base;
  addr_t _phys_base;
  size_t _offset = 0;
  size_t _len;

};

Pmem_allocator* g_pmem_allocator = nullptr;

class Main {
public:
  Main(const vector<string>& pci_id_vector,
       const std::string& pmem_path,
       unsigned duration);
  
  ~Main();

  void run();

  const vector<Component::IBlock_device*>& block_v() const { return _block_v; }
  const std::string pmem_path() const { return _pmem_path; }
  const unsigned duration() const { return _duration; }
  
private:
  void create_block_components(const vector<string>& vs);

  vector<Component::IBlock_device*> _block_v;
  Core::Poller*                     _io_poller;
  std::string                       _pmem_path;
  unsigned                          _duration;
};

Main::Main(const vector<string>& pci_id_vector, const std::string& pmem_path, unsigned duration)
  : _pmem_path(pmem_path), _duration(duration) {
  create_block_components(pci_id_vector);
}

Main::~Main() {
  _io_poller->signal_exit();
  delete _io_poller;

  for (auto& r : _block_v) {
    r->release_ref();
  }
}

int main(int argc, char* argv[]) {
  Main* m = nullptr;
  
  try {
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
      ("pci", po::value<vector<string>>(), "PCIe id for NVMe drive")
      ("threads", po::value<int>(), "# client threads")
      ("chunk_size_in_block", po::value<int>(), "# how many blocks to submit in one async call")
      ("iothreads", po::value<int>(), "# IO threads (default 1)")
      ("rw", "read-write workload")("randwrite", "randomw-write workload")
      ("randread", "random-read workload")
      ("pmem", po::value<std::string>(),"Use persistent memory for main memory buffers")
      ("time", po::value<unsigned>()->default_value(60),"Duration in seconds")
      ("help", "Show help.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
      std::cout << desc;
      return -1;
    }

    if (vm.count("threads")) {
      Options.n_client_threads = vm["threads"].as<int>();
    } else {
      Options.n_client_threads = vm["pci"].as<std::vector<std::string>>().size();
    }

    if (vm.count("iothreads")) Options.n_io_threads = vm["iothreads"].as<int>();

    if (vm.count("chunk_size_in_block"))
      Options.chunk_size_in_block = vm["chunk_size_in_block"].as<int>();

    if (vm.count("rw")) {
      PINF("Using RW 50:50 workload");
      Options.workload = WORKLOAD_RW;
    } else if (vm.count("randwrite")) {
      PINF("Using random write workload");
      Options.workload = WORKLOAD_RAND_WRITE;
    } else {
      PINF("Using random read workload");
      Options.workload = WORKLOAD_RAND_READ;
    }

    if (vm.count("pci")) {
      DPDK::eal_init(256);

      std::string pmem_path;
      if (vm.count("pmem")) {
        pmem_path = vm["pmem"].as<const std::string&>();
        g_pmem_allocator = new Pmem_allocator(pmem_path);
      }

      m = new Main(vm["pci"].as<std::vector<std::string>>(), pmem_path, vm["time"].as<unsigned>());
      m->run();
      delete m;
    } else {
      printf("block-perf [--pci 8b:00.0 --pci 86:00.0 ] --threads 4\n");
      return -1;
    }
  } catch (...) {
    printf("block-perf [--pci 8b:00.0 --pci 86:00.0 ] --threads 4\n");
    return -1;
  }

  /* clean up */
  if(g_pmem_allocator)
    delete g_pmem_allocator;
  
  return 0;
}

class IO_task : public Core::Tasklet {
private:
  static constexpr bool option_DEBUG = true;

public:
  IO_task(Main* m) {  // vector<Component::IBlock_device *> bdv) : _bdv(bdv) {
    _bdv = m->block_v();
    _pmem_path = m->pmem_path();
  }

  void initialize(unsigned core) override {
    unsigned index = core % _bdv.size();
    _block = _bdv[index];
    _block->get_volume_info(_vi);

    if(_vi.block_size != 4096)
      throw General_exception("block device is not in 4K format");
    
    PLOG("IO_task: %p is using IBlock_device %p (%s) %u/%lu, chunk size %u*4KB",
         this, _block, _vi.volume_name, index, _bdv.size(),
         Options.chunk_size_in_block);

    size_t size = Options.chunk_size_in_block * KB(4) + MB(2);
    /* check for persistent memory use */
    if (_pmem_path == "") {
      /* create buffers */
      for (unsigned i = 0; i < NUM_BUFFERS; i++) {
        auto iob =
          _block->allocate_io_buffer(size, KB(4), Component::NUMA_NODE_ANY);
        _buffer_q.enqueue(iob);
      }
    }
    else {
      PLOG("(Experimental) using persistent memory (%s)", _pmem_path.c_str());

      for (unsigned i = 0; i < NUM_BUFFERS; i++) {
        void * vaddr;
        addr_t paddr = 0;
        vaddr = g_pmem_allocator->allocate(size, paddr);
        auto iob = _block->register_memory_for_io(vaddr, paddr, size);

        _buffer_q.enqueue(iob);
      }

      PLOG("IO_task: core(%u) task(%p) using index (%u) pthread (%p)", core,
           this, index, (void*)pthread_self());
    }
    /* start timer after initialization */
    PLOG("Start time stamped.");
    _ts_start = std::chrono::high_resolution_clock::now();
    
  }

  struct memory_pair {
    Component::io_buffer_t iob;
    IO_task* pthis;
  };

  static void release_cb(uint64_t gwid, void* arg0, void* arg1) {
    memory_pair* mp = (memory_pair*)arg0;
    mp->pthis->free_buffer(mp->iob);
    delete mp;
  }

  void do_work(unsigned core) override {
    memory_pair* mp = new memory_pair;
    mp->iob = alloc_buffer();
    mp->pthis = this;

    auto block = genrand64_int64() % (_vi.block_count - Options.chunk_size_in_block);

    switch (Options.workload) {
    case WORKLOAD_RAND_READ:
      _last_gwid = _block->async_read(mp->iob, 0, block, Options.chunk_size_in_block, /* n blocks */
                                      (core % Options.n_io_threads) + START_IO_CORE, release_cb,
                                      (void*)mp);
      _io_count++;
      break;
    case WORKLOAD_RAND_WRITE:
      _last_gwid =
        _block->async_write(mp->iob, 0, block, Options.chunk_size_in_block, /* n blocks */
                            (core % Options.n_io_threads) + START_IO_CORE,
                            release_cb, (void*)mp);
      _io_count++;
      break;
    case WORKLOAD_RW:
      _block->read(mp->iob, 0, block, Options.chunk_size_in_block, /* n blocks */
                   (core % Options.n_io_threads) + START_IO_CORE);

      _block->write(mp->iob, 0, block, Options.chunk_size_in_block, /* n blocks */
                    (core % Options.n_io_threads) + START_IO_CORE);

      free_buffer(mp->iob);
      _io_count += 2;
      break;
    default:
      throw General_exception("unhandled workload");
    }
  }

  void cleanup(unsigned core) override {
    if (_last_gwid) {
      PINF("IO_task: check completion for gwid %lu", _last_gwid);
      _block->check_completion(_last_gwid,
                               (core % Options.n_io_threads) + START_IO_CORE);
    }

    _ts_end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(_ts_end-_ts_start).count() / 1000.0;
    PLOG("End time stamped. Duration %.2g", secs);

    PINF("(%p) %lu operations at throughput %.2f MB/s",
         this,
         _io_count,
         (((_io_count / secs) * 4.0 * Options.chunk_size_in_block)) / 1024.0);
  }

  Component::io_buffer_t alloc_buffer() {
    Component::io_buffer_t iob;
    while (!_buffer_q.dequeue(iob)) {
      cpu_relax();
    }
    return iob;
  }

  void free_buffer(Component::io_buffer_t iob) {
    while (!_buffer_q.enqueue(iob)) {
      cpu_relax();
    }
  }

  size_t io_count() const { return _io_count; }

private:
  static constexpr unsigned NUM_BUFFERS = 1024;  

  Component::VOLUME_INFO            _vi;
  unsigned long                     _io_count = 0;
  std::string                       _pmem_path;
  vector<Component::IBlock_device*> _bdv;
  Component::IBlock_device*         _block;

  Common::Spsc_bounded_lfq_sleeping<Component::io_buffer_t, NUM_BUFFERS>  _buffer_q;
  std::chrono::time_point<std::chrono::system_clock> _ts_start, _ts_end;
  uint64_t _last_gwid = 0;
};

void Main::run() {
  PMAJOR("Using %u client threads", Options.n_client_threads);
  PMAJOR("Using %u IO threads", Options.n_io_threads);

  cpu_mask_t m; /* these are the driving client threads */
  for (unsigned i = 0; i < Options.n_client_threads; i++) {
    m.add_core(i + START_CLIENT_CORE);
  }

  {
    Core::Per_core_tasking<IO_task, typeof(this)> workers(m, this);
    sleep(_duration);
  }
  sleep(1);
  PLOG("Per core tasking complete");
}

void Main::create_block_components(const vector<string>& vs) {

  using namespace Component;
  IBase* comp = load_component("libcomanche-blknvme.so", block_nvme_factory);
  assert(comp);
  IBlock_device_factory* fact = (IBlock_device_factory*)comp->query_interface(IBlock_device_factory::iid());

  cpu_mask_t m;
  for (unsigned i = 0; i < Options.n_io_threads; i++) {
    m.add_core(START_IO_CORE + i);
  }

  _io_poller = new Core::Poller(m);

  for (auto& id : vs) {
    PMAJOR("Attaching to %s (using poller)..", id.c_str());
    IBlock_device* itf = fact->create(id.c_str(), nullptr, _io_poller);
    _block_v.push_back(itf);
  }

  fact->release_ref();
}
