/* note: we do not include component source, only the API definition */
#include <common/utils.h>
#include <common/str_utils.h>
#include <core/task.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <gperftools/profiler.h>

#include <core/dpdk.h>

#define PATH "/mnt/pmem0/"
//#define PATH "/dev/dax0.0"
#define POOL_NAME "test.pool"

#define PMSTORE_PATH "libcomanche-pmstore.so"
#define FILESTORE_PATH "libcomanche-storefile.so"
#define NVMESTORE_PATH "libcomanche-nvmestore.so"
#define ROCKSTORE_PATH "libcomanche-rocksdb.so"
#define DEFAULT_COMPONENT "pmstore"

using namespace Component;

#include "data.h"
#include "exp_put.h"
#include "exp_get.h"

Data * _data;

static Component::IKVStore * g_store;
static void initialize();
static void cleanup();

// required for nvmestore
static Component::IBlock_device  *_block;
static Component::IBlock_allocator *_alloc;
static size_t _nr_blks; //number of blocks from IBlockdev

struct {
  std::string test;
  std::string component;
  unsigned cores;
  unsigned time_secs;
} Options;


int main(int argc, char * argv[])
{
  ProfilerDisable();
  
  namespace po = boost::program_options; 
  po::options_description desc("Options"); 
  desc.add_options()
    ("help", "Show help")
    ("test", po::value<std::string>(), "Test name <all|Put|Get>")
    ("component", po::value<std::string>(), "Implementation selection <pmstore|nvmestore|filestore>")
    ("cores", po::value<int>(), "Number of threads/cores")
    ("time", po::value<int>(), "Duration to run in seconds")
    ;

  try {
    po::variables_map vm; 
    po::store(po::parse_command_line(argc, argv, desc),  vm);

    if(vm.count("help")) {
      std::cout << desc;
      return 0;
    }

    Options.test = vm.count("test") > 0 ? vm["test"].as<std::string>() : "all";
    vm.count("component") > 0 ? vm["component"].as<std::string>() : DEFAULT_COMPONENT;

    if(vm.count("component"))
      Options.component = vm["component"].as<std::string>();
    else
      Options.component = DEFAULT_COMPONENT;
    
    Options.cores  = vm.count("cores") > 0 ? vm["cores"].as<int>() : 1;
    Options.time_secs  = vm.count("time") > 0 ? vm["time"].as<int>() : 4;
  }
  catch (const po::error &ex)
  {
    std::cerr << ex.what() << '\n';
  }

  _data = new Data();
  initialize();

  cpu_mask_t cpus;
  unsigned core = 1;
  for(unsigned core = 0; core < Options.cores; core++)
    cpus.add_core(core);

  ProfilerStart("cpu.profile");

  if(Options.test == "all" || Options.test == "Put") {
    Core::Per_core_tasking<Experiment_Put, Component::IKVStore*> exp(cpus, g_store);
    sleep(Options.time_secs);
  }

  if(Options.test == "all" || Options.test == "Get") {
    Core::Per_core_tasking<Experiment_Get, Component::IKVStore*> exp(cpus, g_store);
    //    sleep(Options.time_secs + 8);
    exp.wait_for_all();
  }

  
  ProfilerStop();
  
  cleanup();
  
  return 0;
}


static void initialize()
{
  Component::IBase * comp;
  
  if(Options.component == "pmstore") {
    comp = Component::load_component(PMSTORE_PATH, Component::pmstore_factory);
  }
  else if(Options.component == "filestore") {
    comp = Component::load_component(FILESTORE_PATH, Component::filestore_factory);
  }
  else if(Options.component == "nvmestore") {

    DPDK::eal_init(512);

    /*
     * init blk devices
     */
#ifdef USE_SPDK_NVME_DEVICE
    
    comp = Component::load_component("libcomanche-blknvme.so",
                                                        Component::block_nvme_factory);

    assert(comp);
    PLOG("Block_device factory loaded OK.");
    IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
    
    cpu_mask_t cpus;
    cpus.add_core(2);

    _block = fact->create("86:00.0", &cpus);

    assert(_block);
    fact->release_ref();
    PINF("Lower block-layer component loaded OK.");

#else
    
    comp = Component::load_component("libcomanche-blkposix.so",
                                                        Component::block_posix_factory);
    assert(comp);
    PLOG("Block_device factory loaded OK.");

    IBlock_device_factory * fact_blk = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
    std::string config_string;
    config_string = "{\"path\":\"";
    //  config_string += "/dev/nvme0n1";1
    config_string += "./blockfile.dat";
    //  config_string += "\"}";
    config_string += "\",\"size_in_blocks\":10000}";
    PLOG("config: %s", config_string.c_str());

    _nr_blks = 10000;

    _block = fact_blk->create(config_string);
    assert(_block);
    fact_blk->release_ref();
    PINF("Block-layer component loaded OK (itf=%p)", _block);
#endif

    /*
     * instantiate block allocator
     */
    comp = load_component("libcomanche-blkalloc-aep.so",
                                  Component::block_allocator_aep_factory);
    assert(comp);
    IBlock_allocator_factory * fact_blk_alloc = static_cast<IBlock_allocator_factory *>
      (comp->query_interface(IBlock_allocator_factory::iid()));

    size_t num_blocks = _nr_blks;
    PLOG("Opening allocator to support %lu blocks", num_blocks);
    _alloc = fact_blk_alloc->open_allocator(
                                  num_blocks,
                                  "nvmestore-blk-allc-1");  
    fact_blk_alloc->release_ref();  

    /*
     * instantialize
     */
    comp = Component::load_component("libcomanche-nvmestore.so",
                                                        Component::nvmestore_factory);
  }
  else if(Options.component == "rockstore") {
    comp = Component::load_component(ROCKSTORE_PATH, Component::rocksdb_factory);
  }
  else throw General_exception("unknown --component option (%s)", Options.component.c_str());

  assert(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  if(Options.component == "nvmestore") {
    g_store = fact->create("owner","name", _block, _alloc ); 
  else{
    g_store = fact->create("owner","name"); }
  }
  fact->release_ref();
}

static void cleanup()
{
  g_store->release_ref();

  if(Options.component == "nvmestore") {
    assert(_alloc);
    assert(_block);
    _alloc->release_ref();
    _block->release_ref();
  }
}



