/* note: we do not include component source, only the API definition */
#include <common/utils.h>
#include <common/str_utils.h>
#include <core/task.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <gperftools/profiler.h>
#define PATH "/mnt/pmem0/"
//#define PATH "/dev/dax0.0"
#define POOL_NAME "test.pool"

using namespace Component;

class Data
{
public:
  static constexpr size_t NUM_ELEMENTS = 10000000;
  static constexpr size_t KEY_LEN = 8;
  static constexpr size_t VAL_LEN = 64;

  struct KV_pair {
    char key[KEY_LEN + 1];
    char value[VAL_LEN + 1];
  };

  KV_pair * _data;

  Data() {
    PLOG("Initializing data....");
    _data = new KV_pair[NUM_ELEMENTS];

    for(size_t i=0;i<NUM_ELEMENTS;i++) {
      auto key = Common::random_string(KEY_LEN);
      auto val = Common::random_string(VAL_LEN);
      strncpy(_data[i].key, key.c_str(), key.length());
      _data[i].key[KEY_LEN] = '\0';
      strncpy(_data[i].value, val.c_str(), val.length());
     _data[i].value[VAL_LEN] = '\0';
    }
    PLOG("Initializing data..OK.");
  }

  ~Data() {
    delete [] _data;
  }

  const char * key(size_t i) const {
    if(i >= NUM_ELEMENTS) throw General_exception("out of bounds");
    return _data[i].key;
  }

  const char * value(size_t i) const {
    if(i >= NUM_ELEMENTS) throw General_exception("out of bounds");
    return _data[i].value;
  }

  size_t value_len() const { return VAL_LEN; }

};

Data * _data;

class Experiment_Put : public Core::Tasklet
{ 
public:

  Experiment_Put(Component::IKVStore * arg) : _store(arg) {
    assert(arg);
  }
  
  void initialize(unsigned core) {
    char poolname[256];
    sprintf(poolname, "Put.pool.%u", core);
    PLOG("Creating pool for worker %u ...", core);
    _pool = _store->create_pool("/mnt/pmem0", poolname, GB(1));
    PLOG("Created pool for worker %u ...OK!", core);
    //    _pool = _store->open_pool("/dev/", "dax0.0");
    ProfilerRegisterThread();
  };
  
  void do_work(unsigned core) {
    if(_first_iter) {
      _start = std::chrono::high_resolution_clock::now();
      _first_iter = false;
    }
    
    i++;
    int rc = _store->put(_pool, _data->key(i), _data->value(i), _data->value_len());
    assert(rc == S_OK);
  }
  
  void cleanup(unsigned core) {
    _end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / 1000.0;
    PINF("*Put* IOPS: %2g", ((double)i) / secs);

    _store->delete_pool(_pool);
  }

private:
  size_t i = 0;
  
  Component::IKVStore *       _store;
  Component::IKVStore::pool_t _pool;
  bool                        _first_iter = true;
  std::chrono::system_clock::time_point _start, _end;
};


static Component::IKVStore * g_store;
static void initialize();
static void cleanup();

struct {
  std::string test;
} Options;


int main(int argc, char * argv[])
{
  ProfilerDisable();

  _data = new Data();
  
  namespace po = boost::program_options; 
  po::options_description desc("Options"); 
  desc.add_options()
    ("help", "Show help"),
    ("test", po::value<std::string>(), "Test name")
    ;

  try {
    po::variables_map vm; 
    po::store(po::parse_command_line(argc, argv, desc),  vm);

    if(vm.count("help")) {
      std::cout << desc;
      return 0;
    }

    if(vm.count("test"))
      Options.test = vm["test"].as<std::string>();
    else
      Options.test = "all";
  }
  catch (...) {
    std::cout << desc;
    return 0;    
  }

  initialize();

  cpu_mask_t cpus;
  //  cpus.set_mask(0xF);
  cpus.add_core(1);
  cpus.add_core(2);
  cpus.add_core(3);
  cpus.add_core(4);

  ProfilerStart("cpu.profile");
  {
    Core::Per_core_tasking<Experiment_Put, Component::IKVStore*> tasking(cpus, g_store);
    sleep(4);
  }
  ProfilerStop();
  
  cleanup();
  
  return 0;
}


static void initialize()
{
  Component::IBase * comp = Component::load_component("/home/danielwaddington/comanche/build/comanche-restricted/src/components/pmstore/libcomanche-pmstore.so", Component::pmstore_factory);

  assert(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  g_store = fact->create("owner","name");  
  fact->release_ref();
}

static void cleanup()
{
  g_store->release_ref();
}



