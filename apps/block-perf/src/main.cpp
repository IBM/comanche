#include <boost/program_options.hpp> 
#include <core/task.h>
#include <core/poller.h>
#include <core/dpdk.h>
#include <common/assert.h>
#include <common/utils.h>
#include <common/rand.h>
#include <common/exceptions.h>
#include <common/spsc_bounded_queue.h>
#include <api/components.h>
#include <api/block_itf.h>
#include <string>
#include <vector>
#include <stdio.h>

using namespace std;

enum {
  WORKLOAD_RAND_READ = 1,
  WORKLOAD_RAND_WRITE = 2,
  WORKLOAD_RW = 5,
};
  
struct {
  unsigned n_client_threads;
  unsigned n_io_threads = 1;
  int workload = WORKLOAD_RAND_READ;
} Options;


#define START_IO_CORE 2
#define START_CLIENT_CORE 6

class Main
{
public:
  Main(const vector<string>& pci_id_vector);
  ~Main();

  void run();
  
private:
  void create_block_components(const vector<string>& vs);

  vector<Component::IBlock_device *> _block_v;
  Core::Poller *                     _io_poller;  
};


Main::Main(const vector<string>& pci_id_vector)
{
  create_block_components(pci_id_vector);
}

Main::~Main()
{
  _io_poller->signal_exit();
  delete _io_poller;
  
  for(auto& r: _block_v) {
    r->release_ref();
  }

}


int main(int argc, char * argv[])
{
  Main * m = nullptr;
  
  try {
    namespace po = boost::program_options; 
    po::options_description desc("Options"); 
    desc.add_options()
      ("pci", po::value< vector<string> >(), "PCIe id for NVMe drive")
      ("threads", po::value<int>(), "# client threads")
      ("iothreads", po::value<int>(), "# IO threads (default 1)")
      ("rw", "read-write workload")
      ("randwrite", "randomw-write workload")
      ("randread", "random-read workload")
      ;
 
    po::variables_map vm; 
    po::store(po::parse_command_line(argc, argv, desc),  vm);

    if(vm.count("threads")) {
      Options.n_client_threads = vm["threads"].as<int>();
    }
    else {
      Options.n_client_threads = vm["pci"].as<std::vector<std::string>>().size();
    }

    if(vm.count("iothreads"))
      Options.n_io_threads = vm["iothreads"].as<int>();
    
    if(vm.count("rw")) {
      PINF("Using RW 50:50 workload");
      Options.workload = WORKLOAD_RW;
    }
    else if(vm.count("randwrite")) {
      PINF("Using random write workload");
      Options.workload = WORKLOAD_RAND_WRITE;
    }
    else {
      PINF("Using random read workload");
      Options.workload = WORKLOAD_RAND_READ;
    }
       
    if(vm.count("pci")) {
      m = new Main(vm["pci"].as<std::vector<std::string>>());
      m->run();
      delete m;
    }
    else {
      printf("block-perf [--pci 8b:00.0 --pci 86:00.0 ] --threads 4\n");
      return -1;
    }
  }
  catch(...) {
    printf("block-perf [--pci 8b:00.0 --pci 86:00.0 ] --threads 4\n");
    return -1;
  }

  return 0;
}

class IO_task : public Core::Tasklet
{
private:
  static constexpr bool option_DEBUG = true;
  
public:
  IO_task(vector<Component::IBlock_device *> bdv) : _bdv(bdv) {
  }
  
  void initialize(unsigned core) override {
    unsigned index = core % _bdv.size();
    _block = _bdv[index];
    _block->get_volume_info(_vi);
    PLOG("IO_task: %p is using IBlock_device %p (%s) %u/%lu", this, _block, _vi.volume_name, index, _bdv.size());

    /* create buffers */
    for(unsigned i=0;i<NUM_BUFFERS;i++) {
      auto iob = _block->allocate_io_buffer(KB(4),KB(4), Component::NUMA_NODE_ANY);
      _buffer_q.enqueue(iob);
    }
    
    PLOG("IO_task: core(%u) task(%p) using index (%u) pthread (%p)", core, this, index, (void*) pthread_self());
    _iob = _block->allocate_io_buffer(KB(4),KB(4), Component::NUMA_NODE_ANY);
  }

  struct memory_pair {
    Component::io_buffer_t iob;
    IO_task* pthis;
  };

  static void release_cb(uint64_t gwid, void * arg0, void* arg1) {
    memory_pair *mp = (memory_pair *)arg0;
    mp->pthis->free_buffer(mp->iob);
    delete mp;
  }
  
  void do_work(unsigned core) override {
    memory_pair *mp = new memory_pair;
    mp->iob = alloc_buffer();
    mp->pthis = this;

    auto block = genrand64_int64() % _vi.block_count;

    switch(Options.workload) {
    case WORKLOAD_RAND_READ:
      _last_gwid = _block->async_read(mp->iob,
                                      0,
                                      block,
                                      1, /* n blocks */
                                      (core % Options.n_io_threads) + START_IO_CORE,
                                      release_cb, (void*) mp);
      _io_count++;
      break;
    case WORKLOAD_RAND_WRITE:
      _last_gwid = _block->async_write(mp->iob,
                                      0,
                                      block,
                                      1, /* n blocks */
                                      (core % Options.n_io_threads) + START_IO_CORE,
                                      release_cb, (void*) mp);
      _io_count++;
      break;
    case WORKLOAD_RW:
      _block->read(mp->iob,
                   0,
                   block,
                   1, /* n blocks */
                   (core % Options.n_io_threads) + START_IO_CORE);
      
      _block->write(mp->iob,
                    0,
                    block,
                    1, /* n blocks */
                    (core % Options.n_io_threads) + START_IO_CORE);

      free_buffer(mp->iob);
      
#if 0
      _last_gwid = _block->async_write(mp->iob,
                                      0,
                                      block,
                                      1, /* n blocks */
                                      (core % Options.n_io_threads) + START_IO_CORE,
                                      release_cb, (void*) mp);
#endif
      _io_count+=2;
      break;
    default:
      throw General_exception("unhandled workload");
    }

  }
  
  void cleanup(unsigned core) override {
    //    _block->check_completion(_last_gwid, (core % Options.n_io_threads) + START_IO_CORE);
    PINF("(%p) completed %lu IOPS %.2f MB/s", this,
         _io_count/10 /* period is 10sec */,
         (_io_count/10 * 4.0f) / 1024.0);
  }

  Component::io_buffer_t alloc_buffer()
  {
    Component::io_buffer_t iob;
    while(!_buffer_q.dequeue(iob)) {
      cpu_relax();
    }
    return iob;
  }

  void free_buffer(Component::io_buffer_t iob) {
    while(!_buffer_q.enqueue(iob)) {
      cpu_relax();
    }
  }

  size_t io_count() const { return _io_count; }
  
private:
  static constexpr unsigned NUM_BUFFERS = 1024;

  Component::VOLUME_INFO             _vi;
  unsigned long                      _io_count = 0;
  vector<Component::IBlock_device *> _bdv;
  Component::IBlock_device *         _block;
  Component::io_buffer_t             _iob;
  Common::Spsc_bounded_lfq_sleeping<Component::io_buffer_t, NUM_BUFFERS> _buffer_q;
  uint64_t _last_gwid;
};

void Main::run()
{
  PMAJOR("Using %u client threads", Options.n_client_threads);
  PMAJOR("Using %u IO threads", Options.n_io_threads);
  
  cpu_mask_t m; /* these are the driving client threads */
  for(unsigned i=0;i<Options.n_client_threads;i++) {
    m.add_core(i + START_CLIENT_CORE);
  }

  //  IO_task iot(_block_v);
  {
    Core::Per_core_tasking<IO_task,typeof(_block_v)> workers(m, _block_v);
    
    sleep(10);
  }
  PLOG("Per core tasking complete");
  sleep(1);
}

void Main::create_block_components(const vector<string>& vs)
{
  DPDK::eal_init(256);
    
  using namespace Component;
  IBase * comp = load_component("libcomanche-blknvme.so", block_nvme_factory);
  assert(comp);
  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  
  cpu_mask_t m;
  for(unsigned i=0;i<Options.n_io_threads;i++) {
    m.add_core(START_IO_CORE + i);
  }
  
  _io_poller = new Core::Poller(m);
  
  for(auto& id: vs) {
    PMAJOR("Attaching to %s (using poller)..", id.c_str());
    IBlock_device * itf = fact->create(id.c_str(), nullptr, _io_poller);
    _block_v.push_back(itf);
  }
  
  fact->release_ref();
}

