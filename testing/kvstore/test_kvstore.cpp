/* set up unit test framework first*/
//#define BOOST_TEST_DYN_LINK  // TODO: not sure if we need this, but keep it around for now
#define BOOST_TEST_MODULE "CorrectnessTest"
#include <boost/test/included/unit_test.hpp>

/* note: we do not include component source, only the API definition */
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <boost/program_options.hpp>
#include <common/utils.h>
#include <common/str_utils.h>
#include <core/task.h>
//#include <chrono>
#include <iostream>
//#include <gperftools/profiler.h>
//
//#define PATH "/mnt/pmem0/"
//#define PATH "/dev/dax0.0"
//#define POOL_NAME "test.pool"
#define FILESTORE_PATH "libcomanche-storefile.so"

using namespace Component;

struct fixture
{
    fixture()
    {
        printf("component created\n");
    }

    ~fixture()
    {
        printf("component destroyed\n");
    }
};

BOOST_AUTO_TEST_SUITE(unit_tests)
BOOST_GLOBAL_FIXTURE(fixture)

BOOST_AUTO_TEST_CASE(load_test)
{
    printf("load_test running\n");

    Component::IKVStore * g_store;
    Component::IBase * comp;

    printf("variables declared\n");

    comp = Component::load_component(FILESTORE_PATH, Component::filestore_factory);
    assert(comp); // TODO: assert message possible here?

    printf("component loaded\n");

    IKVStore_factory * fact = (IKVStore_factory *)comp->query_interface(IKVStore_factory::iid());

    printf("query_interface completed\n");

    g_store = fact->create("owner", "name");  // TODO: what does this do?

    printf("create done\n");

    fact->release_ref();  // TODO: what does this do?

    printf("release_ref done\n");

    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END()

/*
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
    // TODO: is it possible to register all tests and write them out automatically?
    ("test", po::value<std::string>(), "Test name <all|Put|Get>")
    // TODO: is it possible to register all components and write them out automatically?
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
      printf("Put test starting\n");
    Core::Per_core_tasking<Experiment_Put, Component::IKVStore*> exp(cpus, g_store);
    sleep(Options.time_secs);
  }

  if(Options.test == "all" || Options.test == "Get") {
      printf("Get test starting\n");
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
    comp = Component::load_component(NVMESTORE_PATH, Component::nvmestore_factory);
  }
  else if(Options.component == "rockstore") {
    comp = Component::load_component(ROCKSTORE_PATH, Component::rocksdb_factory);
  }
  else throw General_exception("unknown --component option (%s)", Options.component.c_str());

  assert(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  if(Options.component == "nvmestore"){
    g_store = fact->create("owner","name", "81:00.0");
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


*/
