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
#include "exp_get_direct.h"
#include "exp_put_direct.h"
#include "kvstore_perf.h"

ProgramOptions Options;
Data * _data;

static Component::IKVStore * g_store;
static int initialize();
static void cleanup();

int g_argc;
char ** g_argv;

pthread_mutex_t g_write_lock = PTHREAD_MUTEX_INITIALIZER;
boost::program_options::options_description desc("Options"); 

int main(int argc, char * argv[])
{
  ProfilerDisable();
  
  g_argc = argc;
  g_argv = argv;

  namespace po = boost::program_options; 

  try {
    desc.add_options()
      ("help", "Show help")
      ("test", po::value<std::string>(), "Test name <all|put|get|put_direct|get_direct>. Default: all.")
      ("component", po::value<std::string>()->default_value(DEFAULT_COMPONENT), "Implementation selection <filestore|pmstore|dawn|nvmestore|mapstore|hstore>. Default: filestore.")
      ("cores", po::value<std::string>(), "Cores to run tasks on. Supports singles and ranges. Example: a,b,c-d. Default: Core 0.")
      ("path", po::value<std::string>(), "Path of directory for pool. Default: current directory.")
      ("size", po::value<unsigned long long int>(), "Size of pool. Default: 100MB.")
      ("flags", po::value<int>(), "Flags for pool creation. Default: none.")
      ("elements", po::value<int>(), "Number of data elements. Default: 100,000.")
      ("key_length", po::value<unsigned int>(), "Key length of data. Default: 8.")
      ("value_length", po::value<unsigned int>(), "Value length of data. Default: 64.")
      ("bins", po::value<unsigned int>(), "Number of bins for statistics. Default: 100. ")
      ("latency_range_min", po::value<double>(), "Lowest latency bin threshold. Default: 10e-9.")
      ("latency_range_max", po::value<double>(), "Highest latency bin threshold. Default: 10e-3.")
      ("debug_level", po::value<int>(), "Debug level. Default: 0.")
      ("owner", po::value<std::string>(), "Owner name for component registration")
      ("server_address", po::value<std::string>(), "Server address, with port")
      ("device_name", po::value<std::string>(), "Device name")
      ("pci_addr", po::value<std::string>(), "PCI address (e.g. 0b:00.0)")
      ("verbose", "Verbose output")    
      ("summary", "Prints summary statement: most frequent latency bin info per core")
      ;

    po::variables_map vm; 
    po::store(po::parse_command_line(argc, argv, desc),  vm);

    if(vm.count("help")) {
      std::cout << desc;
      return 0;
    }

    Options.test = vm.count("test") > 0 ? vm["test"].as<std::string>() : "all";
    Options.component = vm["component"].as<std::string>();
    Options.cores  = vm.count("cores") > 0 ? vm["cores"].as<std::string>() : "0";
    Options.size = vm.count("size") > 0 ? vm["size"].as<unsigned long long int>() : MB(100);
    Options.flags = vm.count("flags") > 0 ? vm["flags"].as<int>() : Component::IKVStore::FLAGS_SET_SIZE;
    Options.debug_level = vm.count("debug_level") > 0 ? vm["debug_level"].as<int>() : 0;
    Options.owner = vm.count("owner") > 0 ? vm["owner"].as<std::string>() : "name";
    Options.server_address = vm.count("server_address") ? vm["server_address"].as<std::string>() : "127.0.0.1";
    Options.device_name = vm.count("device_name") ? vm["device_name"].as<std::string>() : "unused";

    PLOG("Options.component=%s",Options.component.c_str());
    if(Options.component == "nvmestore") {
      if(!vm.count("pci_addr")) {
        std::cout << "Must specify --pci_addr option for nvmestore component\n";
        return -1;
      }
      Options.pci_addr = vm["pci_addr"].as<std::string>();
    }
    else if(Options.component == "pmstore" || Options.component == "htstore") {
      if(vm.count("path") == 0) {
        std::cout << "Must specify --path option for persistent memory store\n";
        return -1;
      }
    }
    
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
    Options.elements = vm.count("elements") > 0 ? vm["elements"].as<int>() : 100000;
    Options.key_length = vm.count("key_length") > 0 ? vm["key_length"].as<unsigned int>() : 8;
    Options.value_length = vm.count("value_length") > 0 ? vm["value_length"].as<unsigned int>() : 64; 
  }
  catch (const po::error &ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  bool use_direct_memory = Options.component == "dawn";
  _data = new Data(Options.elements, Options.key_length, Options.value_length, use_direct_memory);
  if (initialize() != 0)
    {
      PERR("initialize returned an error. Aborting setup.");
      return 1;
    }

  Options.store = g_store;
  Options.report_file_name = Experiment::create_report(Options);

  cpu_mask_t cpus;

  try
    {
      cpus = Experiment::get_cpu_mask_from_string(Options.cores);
    }
  catch(...)
    {
      PERR("couldn't create CPU mask. Exiting.");
      return 1;
    }

  ProfilerStart("cpu.profile");

  if (Options.test == "all" || Options.test == "put") {
    Core::Per_core_tasking<ExperimentPut, ProgramOptions> exp(cpus, Options);
    exp.wait_for_all();
  }

  if (Options.test == "all" || Options.test == "get") {
    Core::Per_core_tasking<ExperimentGet, ProgramOptions> exp(cpus, Options);

    exp.wait_for_all();
  }

  if (Options.test == "all" || Options.test == "get_direct") {
    Core::Per_core_tasking<ExperimentGetDirect, ProgramOptions> exp(cpus, Options);
    exp.wait_for_all();
  }

  if (Options.test == "all" || Options.test == "put_direct") {
    Core::Per_core_tasking<ExperimentPutDirect, ProgramOptions> exp(cpus, Options);
    exp.wait_for_all();
  } 
  ProfilerStop();
  
  cleanup();
  
  return 0;
}


static int initialize()
{
  Component::IBase * comp;
 
  try
    { 
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
      else if(Options.component == "dawn") {
      
        DECLARE_STATIC_COMPONENT_UUID(dawn_factory, 0xfac66078,0xcb8a,0x4724,0xa454,0xd1,0xd8,0x8d,0xe2,0xdb,0x87);  // TODO: find a better way to register arbitrary components to promote modular use
        comp = Component::load_component(DAWN_PATH, dawn_factory);
      }
      else if (Options.component == "hstore") {
        comp = Component::load_component("libcomanche-hstore.so", Component::hstore_factory);
      }
      else if (Options.component == "mapstore") {
        comp = Component::load_component("libcomanche-storemap.so", Component::mapstore_factory);
      }
      else throw General_exception("unknown --component option (%s)", Options.component.c_str());
    }
  catch(...)
    {
      PERR("error during load_component.");
      return 1;
    }

  if (!comp)
    {
      PERR("comp loaded, but returned invalid value");
      return 1;
    }

  try
    {
      IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

      if(Options.component == "nvmestore") {
        g_store = fact->create("owner","name", Options.pci_addr);
      }
      else if (Options.component == "dawn") {
        g_store = fact->create(Options.debug_level, Options.owner, Options.server_address, Options.device_name);
      }
      else {
        g_store = fact->create("owner", Options.owner);
      }
      fact->release_ref();
    }
  catch(...)
    {
      PERR("factory creation step failed");
      return 1;
    }

  return 0;
}

static void cleanup()
{
  g_store->release_ref();
}



