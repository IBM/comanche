/* note: we do not include component source, only the API definition */
#include <common/utils.h>
#include <common/str_utils.h>
#include <core/task.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>
#include <gperftools/profiler.h>
#define DEFAULT_PATH "/mnt/pmem0/"
//#define PATH "/dev/dax0.0"
#define POOL_NAME "test.pool"

#define PMSTORE_PATH "libcomanche-pmstore.so"
#define FILESTORE_PATH "libcomanche-storefile.so"
#define NVMESTORE_PATH "libcomanche-nvmestore.so"
#define ROCKSTORE_PATH "libcomanche-rocksdb.so"
#define DAWN_PATH "libcomanche-dawn-client.so"
#define DEFAULT_COMPONENT "filestore"

using namespace Component;

#include "data.h"
#include "exp_put.h"
#include "exp_get.h"
#include "exp_put_latency.h"
#include "exp_get_latency.h"
#include "exp_get_direct_latency.h"
#include "exp_put_direct_latency.h"
#include "kvstore_perf.h"

ProgramOptions Options;
Data * _data;

static Component::IKVStore * g_store;
static void initialize();
static void cleanup();

int g_argc;
char ** g_argv;
pthread_mutex_t g_write_lock = PTHREAD_MUTEX_INITIALIZER;

int main(int argc, char * argv[])
{
  ProfilerDisable();
  
    g_argc = argc;
    g_argv = argv;

  namespace po = boost::program_options; 
  po::options_description desc("Options"); 
  desc.add_options()
    ("help", "Show help")
    ("test", po::value<std::string>(), "Test name <all|Put|Get>")
    ("component", po::value<std::string>(), "Implementation selection <pmstore|nvmestore|filestore>")
    ("cores", po::value<int>(), "Number of threads/cores")
    ("time", po::value<int>(), "Duration to run in seconds")
    ("path", po::value<std::string>(), "Path of directory for pool")
    ("size", po::value<unsigned int>(), "Size of pool")
    ("flags", po::value<int>(), "Flags for pool creation")
    ("elements", po::value<unsigned int>(), "Number of data elements")
    ("key_length", po::value<unsigned int>(), "Key length of data")
    ("value_length", po::value<unsigned int>(), "Value length of data")
    ("bins", po::value<unsigned int>(), "Number of bins for statistics")
    ("latency_range_min", po::value<unsigned int>(), "Lowest latency bin threshold")
    ("latency_range_max", po::value<unsigned int>(), "Highest latency bin threshold")
    ("debug_level", po::value<int>(), "Debug level")
    ("owner", po::value<std::string>(), "Owner name for component registration")
    ("server_address", po::value<std::string>(), "server address, with port")
    ("device_name", po::value<std::string>(), "device name")
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
    Options.size = vm.count("size") > 0 ? vm["size"].as<unsigned int>() : MB(100);
    Options.flags = vm.count("flags") > 0 ? vm["flags"].as<int>() : Component::IKVStore::FLAGS_SET_SIZE;
    Options.debug_level = vm.count("debug_level") > 0 ? vm["debug_level"].as<int>() : 0;
    Options.owner = vm.count("owner") > 0 ? vm["owner"].as<std::string>() : "name";
    Options.server_address = vm.count("server_address") ? vm["server_address"].as<std::string>() : "127.0.0.1";
    Options.device_name = vm.count("device_name") ? vm["device_name"].as<std::string>() : "unused";

/*
    if(vm.count("path"))
    {
        Options.path = vm["path"].as<std::string>();
    }
    else 
    {
        Options.path = DEFAULT_PATH;
    }
*/
    Options.elements = vm.count("elements") > 0 ? vm["elements"].as<unsigned int>() : 100000;
    Options.key_length = vm.count("key_length") > 0 ? vm["key_length"].as<unsigned int>() : 8;
    Options.value_length = vm.count("value_length") > 0 ? vm["value_length"].as<unsigned int>() : 64; 
  }
  catch (const po::error &ex)
  {
    std::cerr << ex.what() << '\n';
  }

  bool use_direct_memory = Options.component.compare("dawn_client") == 0;
  _data = new Data(Options.elements, Options.key_length, Options.value_length, use_direct_memory);
  initialize();

  Options.store = g_store;
  Options.report_file_name = Experiment::create_report(Options);

  cpu_mask_t cpus;
  unsigned core = 1;
  for(unsigned core = 0; core < Options.cores; core++)
    cpus.add_core(core);

  ProfilerStart("cpu.profile");

  if(Options.test == "all" || Options.test == "Put") {
    Core::Per_core_tasking<ExperimentPut, ProgramOptions> exp(cpus, Options);
    sleep(Options.time_secs);
  }

  if(Options.test == "all" || Options.test == "Get") {
    Core::Per_core_tasking<ExperimentGet, ProgramOptions> exp(cpus, Options);
    //    sleep(Options.time_secs + 8);
    exp.wait_for_all();
  }

  if (Options.test == "all" || Options.test == "put_latency")
  {
      Core::Per_core_tasking<ExperimentPutLatency, ProgramOptions> exp(cpus, Options);
      exp.wait_for_all();
  }

  if (Options.test == "all" || Options.test == "get_latency")
  {
      Core::Per_core_tasking<ExperimentGetLatency, ProgramOptions> exp(cpus, Options);

      exp.wait_for_all();
  }

  if (Options.test == "all" || Options.test == "get_direct_latency")
  {
      Core::Per_core_tasking<ExperimentGetDirectLatency, ProgramOptions> exp(cpus, Options);
      exp.wait_for_all();
  }

  if (Options.test == "all" || Options.test == "put_direct_latency")
  {
      Core::Per_core_tasking<ExperimentPutDirectLatency, ProgramOptions> exp(cpus, Options);
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
    comp = Component::load_component(NVMESTORE_PATH, Component::nvmestore_factory);
  }
  else if(Options.component == "rockstore") {
    comp = Component::load_component(ROCKSTORE_PATH, Component::rocksdb_factory);
  }
  else if(Options.component == "dawn_client")
  {
    
      DECLARE_STATIC_COMPONENT_UUID(dawn_client_factory, 0xfac66078,0xcb8a,0x4724,0xa454,0xd1,0xd8,0x8d,0xe2,0xdb,0x87);  // TODO: find a better way to register arbitrary components to promote modular use
    comp = Component::load_component(DAWN_PATH, dawn_client_factory);
  }
  else throw General_exception("unknown --component option (%s)", Options.component.c_str());

  assert(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  if(Options.component == "nvmestore"){
    g_store = fact->create("owner","name", "09:00.0");
  }
  else if (Options.component == "dawn_client")
  {
      g_store = fact->create(Options.debug_level, Options.owner, Options.server_address, Options.device_name);
  }
  else{
    g_store = fact->create("owner",Options.owner);
  }
  fact->release_ref();
}

static void cleanup()
{
  g_store->release_ref();
}



