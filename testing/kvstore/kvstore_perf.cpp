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
#define DEFAULT_COMPONENT "filestore"

using namespace Component;

#include "data.h"
#include "exp_put.h"
#include "exp_get.h"
#include "exp_put_latency.h"
#include "exp_get_latency.h"
#include "kvstore_perf.h"

ProgramOptions Options;
Data * _data;

static Component::IKVStore * g_store;
static void initialize();
static void cleanup();

int g_argc;
char ** g_argv;

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
    
  }
  catch (const po::error &ex)
  {
    std::cerr << ex.what() << '\n';
  }

  _data = new Data(Options.elements);
  initialize();

  cpu_mask_t cpus;
  unsigned core = 1;
  for(unsigned core = 0; core < Options.cores; core++)
    cpus.add_core(core);

  ProfilerStart("cpu.profile");

  if(Options.test == "all" || Options.test == "Put") {
    Core::Per_core_tasking<ExperimentPut, Component::IKVStore*> exp(cpus, g_store);
    sleep(Options.time_secs);
  }

  if(Options.test == "all" || Options.test == "Get") {
    Core::Per_core_tasking<ExperimentGet, Component::IKVStore*> exp(cpus, g_store);
    //    sleep(Options.time_secs + 8);
    exp.wait_for_all();
  }

  if (Options.test == "all" || Options.test == "put_latency")
  {
      Core::Per_core_tasking<ExperimentPutLatency, Component::IKVStore*> exp(cpus, g_store);
      exp.wait_for_all();
  }

  if (Options.test == "all" || Options.test == "get_latency")
  {
      Core::Per_core_tasking<ExperimentGetLatency, Component::IKVStore*> exp(cpus, g_store);
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
  else throw General_exception("unknown --component option (%s)", Options.component.c_str());

  assert(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  if(Options.component == "nvmestore"){
    g_store = fact->create("owner","name", "09:00.0");
  }
  else{
    g_store = fact->create("owner","name");
  }
  fact->release_ref();
}

static void cleanup()
{
  g_store->release_ref();
}



