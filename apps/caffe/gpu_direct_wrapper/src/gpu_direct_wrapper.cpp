#include <boost/program_options.hpp> 
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/dump_utils.h>
#include <core/dpdk.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <gdrapi.h>
#include <cuda.h>

using namespace std;

extern "C" void cuda_run_test(Component::IKVStore * _kvstore, \
			      std::string key);

class Main
{
public:
  Main(const string kvstore_owner, string kvstore_name);
  ~Main();

  void run();
  
private:
  void create_kvstore_component(std::string owner, std::string name);

  Component::IKVStore * _kvstore;
};


Main::Main(const std::string kvstore_owner, std::string kvstore_name)
{
  create_kvstore_component(kvstore_owner, kvstore_name);
}

Main::~Main()
{
  _kvstore->release_ref();
}


int main(int argc, char * argv[])
{
  Main * m = nullptr;
  
  try {
    namespace po = boost::program_options; 
    po::options_description desc("Options"); 
    desc.add_options()
      ("owner", po::value< string >(), "Owner for this KVStore")
	  ("name", po::value< string >(), "Name for this KVStore")
      ;
 
    po::variables_map vm; 
    po::store(po::parse_command_line(argc, argv, desc),  vm);

    if(vm.count("owner") && vm.count("name")) {
      m = new Main(vm["owner"].as<std::string>(), vm["name"].as<std::string>());
      m->run();
      delete m;
    }
    else {
      printf("gpu_direct_wrapper_test [--owner KVSTORE_OWNER --name KVSTORE_NAME ]\n");
      return -1;
    }
  }
  catch(...) {
    PERR("unhandled exception!");
    return -1;
  }

  return 0;
}

void Main::run()
{
  /* write some data onto device for testing */
  Component::IKVStore::pool_t pool;

  try {
	  pool = _kvstore->create_pool("./", "test1.pool", MB(32));
  }
  catch(...) {
	  pool = _kvstore->open_pool("./", "test1.pool");
  }
  std::string key0 = "MyKey0";
  std::string key1 = "MyKey1";
  std::string str_value = "Hello World!";
  str_value.resize(MB(2));
  _kvstore->put(pool, key0, str_value.c_str(), str_value.length());
  _kvstore->put(pool, key1, str_value.c_str(), str_value.length());
  void * value = nullptr;
  size_t value_len = 0;
  _kvstore->get(pool, key0, value, value_len);

  PINF("Value=(%.50s) %lu", ((char*) value), value_len);
  void * tmp_addr = malloc(value_len);
  _kvstore->get_direct(pool, key0, tmp_addr, value_len);
  PINF("Value=(%.50s) %lu", ((char*) tmp_addr), value_len);
  cuda_run_test(_kvstore, key0);
}

void Main::create_kvstore_component(std::string owner, std::string name)
{
  using namespace Component;
  IBase * comp = load_component("libcomanche-storefile.so", filestore_factory);
  assert(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  _kvstore = fact->create(owner, name);
  fact->release_ref();
}

