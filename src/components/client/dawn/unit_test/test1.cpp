/* note: we do not include component source, only the API definition */
#include <boost/program_options.hpp>
#include <iostream>
#include <gtest/gtest.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <common/str_utils.h>
#include <core/dpdk.h>
#include <sys/mman.h>

#include <chrono> /* milliseconds */
#include <thread> /* this_thread::sleep_for */

//#define TEST_BASIC_PUT_AND_GET
//#define TEST_PUT_DIRECT_0
//#define TEST_PUT_DIRECT_1
#define TEST_PERF_SMALL_PUT
//#define TEST_PERF_SMALL_GET
//#define TEST_PERF_SMALL_PUT_DIRECT
//#define TEST_PERF_LARGE_PUT_DIRECT
//#define TEST_PERF_LARGE_GET_DIRECT


struct
{
  std::string addr;
  std::string pool;
  std::string device;
  unsigned debug_level;
} Options;


using namespace Component;


namespace {

// The fixture for testing class Foo.
class Dawn_client_test : public ::testing::Test {

 protected:

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:
  
  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }
  
  // Objects declared here can be used by all tests in the test case
  static Component::IKVStore * _dawn;
};

Component::IKVStore * Dawn_client_test::_dawn;


DECLARE_STATIC_COMPONENT_UUID(dawn_client, 0x2f666078,0xcb8a,0x4724,0xa454,0xd1,0xd8,0x8d,0xe2,0xdb,0x87);
DECLARE_STATIC_COMPONENT_UUID(dawn_client_factory, 0xfac66078,0xcb8a,0x4724,0xa454,0xd1,0xd8,0x8d,0xe2,0xdb,0x87);

TEST_F(Dawn_client_test, LibInstantiate)
{
}

TEST_F(Dawn_client_test, Instantiate)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-dawn-client.so", dawn_client_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  _dawn = fact->create(Options.debug_level,
                       "dwaddington",
                       Options.addr,
                       Options.device);
  ASSERT_TRUE(_dawn);

  fact->release_ref();
}

#ifdef TEST_BASIC_PUT_AND_GET
TEST_F(Dawn_client_test, BasicPutAndGet)
{
  ASSERT_TRUE(_dawn);
  int rc;
  
  auto pool = _dawn->create_pool("/mnt/pmem0/dawn",
                                 Options.pool.c_str(),
                                 MB(8));

  std::string value = "Hello! Value";  // 12 chars
  void * pv;
  for(unsigned i=0;i<10;i++) {
    rc = _dawn->put(pool, "key0", value.c_str(), value.length());
    PINF("put response:%d", rc);
    ASSERT_TRUE(rc == S_OK || rc == -2);
    
    pv = nullptr;
    size_t pv_len = 0;
    PINF("performing 'get' to retrieve what was put..");
    rc = _dawn->get(pool, "key0", pv, pv_len);
    PINF("get response:%d (%s) len:%lu", rc, (char*) pv, pv_len);
    ASSERT_TRUE(rc == S_OK);
    ASSERT_TRUE(strncmp((char*) pv, value.c_str(), value.length()) == 0);
  }

  _dawn->delete_pool(pool);
  free(pv);
  PLOG("BasicPutAndGet OK!");
}
#endif

#ifdef TEST_PERF_SMALL_PUT
TEST_F(Dawn_client_test, PerfSmallPut)
{
  ASSERT_TRUE(_dawn);
  int rc;
  
  auto pool = _dawn->create_pool("/mnt/pmem0/dawn",
                                 Options.pool.c_str(),
                                 GB(4));

  static constexpr unsigned long ITERATIONS = 1000000;
  static constexpr unsigned long VALUE_SIZE = 32;
  static constexpr unsigned long KEY_SIZE = 8;

  
  struct record_t {
    std::string key;
    char value[VALUE_SIZE];
  };

  record_t * data = (record_t *) malloc(sizeof(record_t) * ITERATIONS);
  ASSERT_FALSE(data == nullptr);

  /* set up data */
  for(unsigned long i=0;i<ITERATIONS;i++) {
    auto val = Common::random_string(VALUE_SIZE);
    auto key = Common::random_string(KEY_SIZE);
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }

  auto start = std::chrono::high_resolution_clock::now();

  for(unsigned long i=0;i<ITERATIONS;i++) {
    rc = _dawn->put(pool,
                    data[i].key,
                    data[i].value, VALUE_SIZE);
    ASSERT_TRUE(rc == S_OK || rc == -2);    
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PMAJOR("PerfSmallPut Ops/Sec: %lu", static_cast<unsigned long>( ITERATIONS / secs ));

  ::free(data);
  
  _dawn->delete_pool(pool);
}
#endif


#ifdef TEST_PERF_SMALL_PUT_DIRECT
TEST_F(Dawn_client_test, PerfSmallPutDirect)
{
  int rc;

    /* open or create pool */
  Component::IKVStore::pool_t pool = _dawn->open_pool("/mnt/pmem0/dawn",
                                                      Options.pool.c_str(),
                                                      0);

  if(pool == Component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = _dawn->create_pool("/mnt/pmem0/dawn",
                              Options.pool.c_str(),
                              GB(1));
  }

  static constexpr unsigned long ITERATIONS = 1000000;
  static constexpr unsigned long VALUE_SIZE = 32;
  static constexpr unsigned long KEY_SIZE = 8;

  
  struct record_t {
    std::string key;
    char value[VALUE_SIZE];
  };

  size_t data_size = sizeof(record_t) * ITERATIONS;
  record_t * data = (record_t *) aligned_alloc(MiB(2), data_size);
  madvise(data, data_size, MADV_HUGEPAGE);

  ASSERT_FALSE(data == nullptr);
  auto handle = _dawn->register_direct_memory(data, data_size); /* register whole region */

  /* set up data */
  for(unsigned long i=0;i<ITERATIONS;i++) {
    auto val = Common::random_string(VALUE_SIZE);
    data[i].key = Common::random_string(KEY_SIZE);
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }

  auto start = std::chrono::high_resolution_clock::now();

  for(unsigned long i=0;i<ITERATIONS;i++) {
    /* different value each iteration; tests memory region registration */
    rc = _dawn->put_direct(pool,
                           data[i].key,
                           data[i].value,
                           VALUE_SIZE,
                           handle); /* pass handle from memory registration */
    ASSERT_TRUE(rc == S_OK || rc == -6);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PMAJOR("PerfSmallPutDirect Ops/Sec: %lu", static_cast<unsigned long>( ITERATIONS / secs ));

  _dawn->unregister_direct_memory(handle);
  
  _dawn->delete_pool(pool);
}
#endif


#ifdef TEST_PERF_LARGE_PUT_DIRECT
TEST_F(Dawn_client_test, PerfLargePutDirect)
{
  int rc;

  /* open or create pool */
  Component::IKVStore::pool_t pool = _dawn->open_pool("/mnt/pmem0/dawn",
                                                      Options.pool.c_str(),
                                                      0);

  if(pool == Component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = _dawn->create_pool("/mnt/pmem0/dawn",
                              Options.pool.c_str(),
                              GB(8));

                             //    virtual pool_t create_pool(const std::string& path,
                             // const std::string& name,
                             // const size_t size,
                             // unsigned int flags = 0,
                             // uint64_t expected_obj_count = 0) = 0;
  }

  

  static constexpr unsigned long PER_ITERATION = 4;
  static constexpr unsigned long ITERATIONS = 100;
  static constexpr unsigned long VALUE_SIZE = MB(512);
  static constexpr unsigned long KEY_SIZE = 16;

  
  struct record_t {
    std::string key;
    char value[VALUE_SIZE];
  };

  PLOG("Allocating buffer with test data ...");
  size_t data_size = sizeof(record_t) * PER_ITERATION;
  record_t * data = (record_t *) aligned_alloc(MiB(2), data_size);
  madvise(data, data_size, MADV_HUGEPAGE);

  ASSERT_FALSE(data == nullptr);
  auto handle = _dawn->register_direct_memory(data, data_size); /* register whole region */

  PLOG("Filling data...");
  /* set up data */
  for(unsigned long i=0;i<PER_ITERATION;i++) {
    auto val = Common::random_string(VALUE_SIZE);
    auto key = Common::random_string(KEY_SIZE);
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }
  PLOG("Starting put operation...");

  auto start = std::chrono::high_resolution_clock::now();

  for(unsigned long i=0;i<(ITERATIONS*PER_ITERATION);i++) {
    /* different value each iteration; tests memory region registration */
    rc = _dawn->put_direct(pool,
                           data[i%PER_ITERATION].key,
                           data[i%PER_ITERATION].value,
                           VALUE_SIZE,
                           handle); /* pass handle from memory registration */ 
    ASSERT_TRUE(rc == S_OK || rc == -6);    
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PINF("PerfLargePutDirect Throughput: %.2f MiB/sec",
       (( PER_ITERATION * ITERATIONS * VALUE_SIZE ) / secs )/(1024.0 * 1024));

  _dawn->close_pool(pool);
  
  _dawn->unregister_direct_memory(handle);
  

}
#endif


#ifdef TEST_PERF_LARGE_GET_DIRECT
TEST_F(Dawn_client_test, PerfLargeGettDirect)
{
  int rc;
  
  auto pool = _dawn->create_pool("/mnt/pmem0/dawn",
                                 Options.pool.c_str(),
                                 GB(8));

  static constexpr unsigned long PER_ITERATION = 8;
  static constexpr unsigned long ITERATIONS = 20;
  static constexpr unsigned long VALUE_SIZE = MB(64);
  static constexpr unsigned long KEY_SIZE = 16;

  
  struct record_t {
    std::string key;
    char value[VALUE_SIZE];
  };

  PLOG("Allocating buffer with test data ...");
  size_t data_size = sizeof(record_t) * PER_ITERATION;
  record_t * data = (record_t *) aligned_alloc(MiB(2), data_size);
  madvise(data, data_size, MADV_HUGEPAGE);

  ASSERT_FALSE(data == nullptr);
  auto handle = _dawn->register_direct_memory(data, data_size); /* register whole region */

  PLOG("Filling data...");
  /* set up data */
  for(unsigned long i=0;i<PER_ITERATION;i++) {
    auto val = Common::random_string(VALUE_SIZE);
    auto key = Common::random_string(KEY_SIZE);
    data[i].key = key;
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }
  PLOG("Starting PUT operation...");

  auto start = std::chrono::high_resolution_clock::now();

  for(unsigned long i=0;i<(ITERATIONS*PER_ITERATION);i++) {
    /* different value each iteration; tests memory region registration */
    rc = _dawn->put_direct(pool,
                           data[i%PER_ITERATION].key,
                           data[i%PER_ITERATION].value,
                           VALUE_SIZE,
                           handle); /* pass handle from memory registration */ 
    ASSERT_TRUE(rc == S_OK || rc == -6);    
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PINF("PerfLargePutDirect Throughput: %.2f MiB/sec", (( PER_ITERATION * ITERATIONS * VALUE_SIZE ) / secs )/(1024.0 * 1024));

  PINF("Starting .. get phase");

  start = std::chrono::high_resolution_clock::now();
  
  /* perform get phase */
  for(unsigned long i=0;i<(ITERATIONS*PER_ITERATION);i++) {
    size_t value_len = VALUE_SIZE;
    rc = _dawn->get_direct(pool,
                           data[i%PER_ITERATION].key,
                           data[i%PER_ITERATION].value,
                           value_len,
                           handle);
    ASSERT_TRUE(rc == S_OK); // || rc == -6);
  }

  end = std::chrono::high_resolution_clock::now();
  secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PINF("PerfLargeGetDirect Throughput: %.2f MiB/sec", (( PER_ITERATION * ITERATIONS * VALUE_SIZE ) / secs )/(1024.0 * 1024));

  _dawn->close_pool(pool);
  
  _dawn->unregister_direct_memory(handle);
}
#endif



#ifdef TEST_PERF_SMALL_GET
TEST_F(Dawn_client_test, PerfSmallGetDirect)
{
  int rc;
  
  auto pool = _dawn->create_pool("/mnt/pmem0/dawn",
                                 Options.pool.c_str(),
                                 GB(8));

  static constexpr unsigned long PER_ITERATION = 8;
  static constexpr unsigned long ITERATIONS = 100000;
  static constexpr unsigned long VALUE_SIZE = 32;
  static constexpr unsigned long KEY_SIZE = 8;

  
  struct record_t {
    std::string key;
    char value[VALUE_SIZE];
  };

  PLOG("Allocating buffer with test data ...");
  size_t data_size = sizeof(record_t) * PER_ITERATION;
  record_t * data = (record_t *) aligned_alloc(MiB(2), data_size);
  madvise(data, data_size, MADV_HUGEPAGE);

  ASSERT_FALSE(data == nullptr);
  auto handle = _dawn->register_direct_memory(data, data_size); /* register whole region */

  PLOG("Filling data...");
  /* set up data */
  for(unsigned long i=0;i<PER_ITERATION;i++) {
    auto val = Common::random_string(VALUE_SIZE);
    auto key = Common::random_string(KEY_SIZE);
    data[i].key = key;
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }
  PLOG("Starting PUT operation...");

  auto start = std::chrono::high_resolution_clock::now();

  for(unsigned long i=0;i<(ITERATIONS*PER_ITERATION);i++) {
    /* different value each iteration; tests memory region registration */
    rc = _dawn->put_direct(pool,
                           data[i%PER_ITERATION].key,
                           data[i%PER_ITERATION].value,
                           VALUE_SIZE,
                           handle); /* pass handle from memory registration */ 
    ASSERT_TRUE(rc == S_OK || rc == -2);    
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PINF("PerfSmallGet Prep-Throughput: %.2f MiB/sec", (( PER_ITERATION * ITERATIONS * VALUE_SIZE ) / secs )/(1024.0 * 1024));

  PINF("Starting .. get phase");

  start = std::chrono::high_resolution_clock::now();
  
  /* perform get phase */
  for(unsigned long i=0;i<(ITERATIONS*PER_ITERATION);i++) {
    size_t value_len = VALUE_SIZE;
    rc = _dawn->get_direct(pool,
                           data[i%PER_ITERATION].key,
                           data[i%PER_ITERATION].value,
                           value_len,
                           handle);
    ASSERT_TRUE(rc == S_OK); // || rc == -6);
  }

  end = std::chrono::high_resolution_clock::now();
  secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PINF("PerfSmallGet Throughput: %.2f MiB/sec", (( PER_ITERATION * ITERATIONS * VALUE_SIZE ) / secs )/(1024.0 * 1024));

  _dawn->close_pool(pool);
  
  _dawn->unregister_direct_memory(handle);
}
#endif


#ifdef TEST_PUT_DIRECT_0
TEST_F(Dawn_client_test, PutDirect0)
{
  auto pool = _dawn->create_pool("/mnt/pmem0/dawn",
                                 "test_pd_8MB",
                                 MB(8));  
  
  ASSERT_TRUE(pool > 0);

  std::string key = "PUT_DIRECT0_key";
  size_t value_len = 32;
  void * value = aligned_alloc(4096, value_len);
  ASSERT_TRUE(value);
  
  memset(value, 'x', value_len);
  ((char*) value)[0] = 'a';
  //  strcpy((char*) value, "This_is_a_value_that_is_put_directly");
  
  ASSERT_TRUE(_dawn->register_direct_memory(value, value_len) == S_OK);

  auto rc = _dawn->put_direct(pool, key.c_str(), key.length(), value, value_len);
  ASSERT_FALSE(rc == Component::IKVStore::E_POOL_NOT_FOUND);
  ASSERT_TRUE(rc == S_OK);

  ASSERT_NO_THROW(_dawn->delete_pool(pool));
  ASSERT_TRUE(_dawn->unregister_direct_memory(value) == S_OK);
}
#endif


#ifdef TEST_PUT_DIRECT_1
TEST_F(Dawn_client_test, PutDirectLarge)
{
  auto pool = _dawn->create_pool("/mnt/pmem0/dawn",
                                 "bigPool4G",
                                 GB(4));  

  std::string key = "PUT_DIRECT1_key";
  size_t value_len = MB(256);
  void * value = aligned_alloc(4096, value_len);
  ASSERT_TRUE(value);
  
  memset(value, 'x', value_len);
  ((char*) value)[0] = 'a';
  //  strcpy((char*) value, "This_is_a_value_that_is_put_directly");
  
  ASSERT_TRUE(_dawn->register_direct_memory(value, value_len) == S_OK);
  
  ASSERT_TRUE(_dawn->put_direct(pool, key.c_str(), key.length(), value, value_len) == S_OK);

  ASSERT_TRUE(pool > 0);
  ASSERT_NO_THROW(_dawn->delete_pool(pool));
  ASSERT_TRUE(_dawn->unregister_direct_memory(value) == S_OK);
}
#endif

TEST_F(Dawn_client_test, Release)
{
  PLOG("Releasing instance...");
  
  /* release instance */
  _dawn->release_ref();
}


} // namespace


int main(int argc, char **argv) {

  //#  option_addr = (argc > 1) ? argv[1] : "10.0.0.41:11911";

  try {
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
      ("help", "Show help")
      ("debug", po::value<unsigned>()->default_value(0), "Debug level 0-3")
      ("server-addr", po::value<std::string>()->default_value("10.0.0.21:11911"), "Server address IP:PORT")
      ("device", po::value<std::string>()->default_value("mlx5_0"), "Network device (e.g., mlx5_0)")
      ("pool", po::value<std::string>()->default_value("myPool"), "Pool name")
      ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc),  vm);

    if(vm.count("help") > 0) {
      std::cout << desc;
      return -1;
    }

    Options.addr = vm["server-addr"].as<std::string>();
    Options.debug_level = vm["debug"].as<unsigned>();
    Options.pool = vm["pool"].as<std::string>();
    Options.device = vm["device"].as<std::string>();   
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }
  catch(...) {
    PLOG("bad command line option configuration");
    return -1;
  }

  return 0;
}
