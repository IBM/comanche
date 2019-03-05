/* note: we do not include component source, only the API definition */
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <common/cpu.h>
#include <common/str_utils.h>
#include <core/dpdk.h>
#include <core/task.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <boost/program_options.hpp>
#include <iostream>

#include <chrono> /* milliseconds */
#include <thread> /* this_thread::sleep_for */

//#define TEST_PERF_SMALL_PUT
//#define TEST_PERF_SMALL_GET_DIRECT
//#define TEST_PERF_LARGE_PUT_DIRECT
//#define TEST_PERF_LARGE_GET_DIRECT

//#define TEST_SCALE_IOPS

// #define TEST_PERF_SMALL_PUT_DIRECT

struct {
  std::string addr;
  std::string pool;
  std::string device;
  unsigned    debug_level;
  unsigned    base_core;
} Options;

Component::IKVStore_factory *fact;

using namespace Component;

namespace
{
// The fixture for testing class Foo.
class Dawn_client_test : public ::testing::Test {
 protected:
  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp()
  {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown()
  {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Objects declared here can be used by all tests in the test case
  static Component::IKVStore *_dawn;
};

Component::IKVStore *Dawn_client_test::_dawn;

DECLARE_STATIC_COMPONENT_UUID(dawn_client,
                              0x2f666078,
                              0xcb8a,
                              0x4724,
                              0xa454,
                              0xd1,
                              0xd8,
                              0x8d,
                              0xe2,
                              0xdb,
                              0x87);
DECLARE_STATIC_COMPONENT_UUID(dawn_client_factory,
                              0xfac66078,
                              0xcb8a,
                              0x4724,
                              0xa454,
                              0xd1,
                              0xd8,
                              0x8d,
                              0xe2,
                              0xdb,
                              0x87);

void basic_test(IKVStore *kv, unsigned shard)
{
  int  rc;
  std::stringstream ss;
  ss << Options.pool << shard;
  std::string pool_name = ss.str();
  auto pool = kv->create_pool(pool_name, MB(8));

  std::string value = "Hello! Value";  // 12 chars

  void *pv;
  for (unsigned i = 0; i < 10; i++) {
    std::string key = Common::random_string(8);
    rc              = kv->put(pool, key.c_str(), value.c_str(), value.length());
    ASSERT_TRUE(rc == S_OK || rc == -2);

    pv            = nullptr;
    size_t pv_len = 0;
    rc            = kv->get(pool, key.c_str(), pv, pv_len);
    ASSERT_TRUE(rc == S_OK);
    ASSERT_TRUE(strncmp((char *) pv, value.c_str(), value.length()) == 0);
  }

  kv->close_pool(pool);
  ASSERT_TRUE(kv->delete_pool(pool_name) == S_OK);
  free(pv);
}


TEST_F(Dawn_client_test, SessionControl)
{
  /* create object instance through factory */
  Component::IBase *comp = Component::load_component(
      "libcomanche-dawn-client.so", dawn_client_factory);

  ASSERT_TRUE(comp);
  fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  auto dawn = fact->create(Options.debug_level, "dwaddington", Options.addr,
                           Options.device);
  ASSERT_TRUE(dawn);
  dawn->release_ref();

  auto dawn2 = fact->create(Options.debug_level, "dwaddington", Options.addr,
                            Options.device);
  ASSERT_TRUE(dawn2);
  auto dawn3 = fact->create(Options.debug_level, "dwaddington", Options.addr,
                            Options.device);
  ASSERT_TRUE(dawn3);

  basic_test(dawn2, 0);
  basic_test(dawn3, 1);

  dawn2->release_ref();
  dawn3->release_ref();

  fact->release_ref();
}



TEST_F(Dawn_client_test, Instantiate)
{
  PMAJOR("Running Instantiate...");
  /* create object instance through factory */
  Component::IBase *comp = Component::load_component(
      "libcomanche-dawn-client.so", dawn_client_factory);

  ASSERT_TRUE(comp);
  fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  _dawn = fact->create(Options.debug_level, "dwaddington",
                       Options.addr,
                       Options.device);
  ASSERT_TRUE(_dawn);

  fact->release_ref();
}
  
TEST_F(Dawn_client_test, OpenCloseDelete)
{
  PMAJOR("Running OpenCloseDelete...");
  using namespace Component;
  IKVStore::pool_t pool, pool2, pool3;

  const std::string poolname = Options.pool + "/OpenCloseDelete";
  ASSERT_TRUE((pool = _dawn->create_pool(poolname, GB(1))) != IKVStore::POOL_ERROR);
  ASSERT_FALSE(pool  == IKVStore::POOL_ERROR);
  ASSERT_TRUE(_dawn->close_pool(pool) == S_OK);

  /* pool already exists */
  ASSERT_TRUE(_dawn->create_pool(poolname, GB(1), IKVStore::FLAGS_CREATE_ONLY) == IKVStore::POOL_ERROR);

  /* open two handles to the same pool + create with implicit open */
  ASSERT_TRUE((pool = _dawn->create_pool(poolname, GB(1))) != IKVStore::POOL_ERROR);
  ASSERT_TRUE((pool2 = _dawn->open_pool(poolname)) != IKVStore::POOL_ERROR);

  /* try delete open pool */
  ASSERT_TRUE(_dawn->delete_pool(poolname) == IKVStore::E_ALREADY_OPEN);

  /* open another */
  ASSERT_TRUE((pool3 = _dawn->open_pool(poolname)) != IKVStore::POOL_ERROR);

  /* close two */
  ASSERT_TRUE(_dawn->close_pool(pool) == S_OK);
  ASSERT_TRUE(_dawn->close_pool(pool2) == S_OK);

  /* try to delete open pool */
  ASSERT_TRUE(_dawn->delete_pool(poolname) == IKVStore::E_ALREADY_OPEN);
  ASSERT_TRUE(_dawn->close_pool(pool3) == S_OK);

  /* ok, now we can delete */
  ASSERT_TRUE(_dawn->delete_pool(poolname) == S_OK);
  PLOG("OpenCloseDelete Test OK");
}

TEST_F(Dawn_client_test, GetNotExist)
{
  PMAJOR("Running PutGet...");
  ASSERT_TRUE(_dawn);
  int rc;

  const std::string poolname = Options.pool + "/GetNotExist";

  auto pool = _dawn->open_pool(poolname, 0);

  if (pool == Component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = _dawn->create_pool(poolname, GB(1));
  }

  void *      pv;
  size_t      pv_len = 0;
  PINF("performing 'get' to retrieve non existing..");
  rc = _dawn->get(pool, "key0", pv, pv_len);
  PINF("get response:%d (%s) len:%lu", rc, (char *) pv, pv_len);
  //  ASSERT_TRUE(rc == S_OK);
  _dawn->close_pool(pool);
  // ASSERT_TRUE(strncmp((char *) pv, value.c_str(), value.length()) == 0);

  _dawn->delete_pool(poolname);
  free(pv);
  PLOG("GetNotExist OK!");
}

TEST_F(Dawn_client_test, BasicPutAndGet)
{
  PMAJOR("Running BasicPutGet...");
  ASSERT_TRUE(_dawn);
  int rc;

  auto pool =
    _dawn->create_pool(Options.pool, MB(8));

  std::string value = "Hello! Value";  // 12 chars
  void *      pv;
  for (unsigned i = 0; i < 10; i++) {
    rc = _dawn->put(pool, "key0", value.c_str(), value.length());
    PINF("put response:%d", rc);
    ASSERT_TRUE(rc == S_OK || rc == -2);

    pv            = nullptr;
    size_t pv_len = 0;
    PINF("performing 'get' to retrieve what was put..");
    rc = _dawn->get(pool, "key0", pv, pv_len);
    PINF("get response:%d (%s) len:%lu", rc, (char *) pv, pv_len);
    ASSERT_TRUE(rc == S_OK);
    ASSERT_TRUE(strncmp((char *) pv, value.c_str(), value.length()) == 0);
  }

  _dawn->delete_pool(Options.pool);
  free(pv);
  PLOG("BasicPutAndGet OK!");
}


#ifdef TEST_SCALE_IOPS

struct record_t {
  std::string key;
  char        value[32];
};

std::mutex    _iops_lock;
static double _iops = 0.0;

class IOPS_task : public Core::Tasklet {
 public:
  static constexpr unsigned long ITERATIONS = 1000000;
  static constexpr unsigned long VALUE_SIZE = 32;
  static constexpr unsigned long KEY_SIZE   = 8;

  IOPS_task(unsigned arg) {}

  virtual void initialize(unsigned core) override
  {
    _store = fact->create(Options.debug_level, "dwaddington", Options.addr,
                          Options.device);

    char poolname[64];
    sprintf(poolname, "/dev/dax0.%u", core);

    _pool = _store->create_pool(poolname, GiB(1));

    _data = (record_t *) malloc(sizeof(record_t) * ITERATIONS);
    ASSERT_FALSE(_data == nullptr);

    PLOG("Setting up data worker: %u", core);

    /* set up data */
    for (unsigned long i = 0; i < ITERATIONS; i++) {
      auto val     = Common::random_string(VALUE_SIZE);
      _data[i].key = Common::random_string(KEY_SIZE);
      memcpy(_data[i].value, val.c_str(), VALUE_SIZE);
    }

    _ready_flag = true;
    _start_time = std::chrono::high_resolution_clock::now();
  }

  virtual bool do_work(unsigned core) override
  {
    if (_iterations == 0) PLOG("Starting worker: %u", core);

    _iterations++;
    status_t rc = _store->put(_pool, _data[_iterations].key,
                              _data[_iterations].value, VALUE_SIZE);

    if (rc != S_OK) throw General_exception("put operation failed:rc=%d", rc);

    assert(rc == S_OK);

    if (_iterations > ITERATIONS) {
      _end_time = std::chrono::high_resolution_clock::now();
      PLOG("Worker: %u complete", core);
      return false;
    }
    return true;
  }

  virtual void cleanup(unsigned core) override
  {
    PLOG("Cleanup %u", core);
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
                      _end_time - _start_time)
                      .count() /
                  1000.0;
    _iops_lock.lock();
    auto iops = ((double) ITERATIONS) / secs;
    PLOG("%f iops (core=%u)", iops, core);
    _iops += iops;
    _iops_lock.unlock();
    _store->close_pool(_pool);
    _store->release_ref();
  }

  virtual bool ready() override { return _ready_flag; }

 private:
  std::chrono::high_resolution_clock::time_point _start_time, _end_time;
  bool                                           _ready_flag = false;
  unsigned long                                  _iterations = 0;
  Component::IKVStore *                          _store;
  record_t *                                     _data;
  Component::IKVStore::pool_t                    _pool;
};

TEST_F(Dawn_client_test, PerfScaleIops)
{
  PMAJOR("Running PerfScaleIops...");
  static constexpr unsigned NUM_CORES = 8;
  cpu_mask_t                mask;
  for (unsigned c = 0; c < NUM_CORES; c++) mask.add_core(c + Options.base_core);
  {
    Core::Per_core_tasking<IOPS_task, unsigned> t(mask, 11911);
    t.wait_for_all();
  }
  PMAJOR("Aggregate IOPS: %2g", _iops);
}
#endif

#ifdef TEST_PERF_SMALL_PUT
TEST_F(Dawn_client_test, PerfSmallPut)
{
  PMAJOR("Running SmallPut...");
  ASSERT_TRUE(_dawn);
  int rc;

  const std::string poolname = Options.pool + "/PerfSmallPut";
  auto pool =
    _dawn->create_pool(poolname, GB(4));

  static constexpr unsigned long ITERATIONS = 1000000;
  static constexpr unsigned long VALUE_SIZE = 32;
  static constexpr unsigned long KEY_SIZE   = 8;

  struct record_t {
    std::string key;
    char        value[VALUE_SIZE];
  };

  record_t *data = (record_t *) malloc(sizeof(record_t) * ITERATIONS);
  ASSERT_FALSE(data == nullptr);

  /* set up data */
  for (unsigned long i = 0; i < ITERATIONS; i++) {
    auto val    = Common::random_string(VALUE_SIZE);
    data[i].key = Common::random_string(KEY_SIZE);
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned long i = 0; i < ITERATIONS; i++) {
    rc = _dawn->put(pool, data[i].key, data[i].value, VALUE_SIZE);
    ASSERT_TRUE(rc == S_OK || rc == -2);
  }

  auto end  = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count() /
              1000.0;
  PMAJOR("PerfSmallPut Ops/Sec: %lu",
         static_cast<unsigned long>(ITERATIONS / secs));

  ::free(data);

  _dawn->close_pool(pool);
  _dawn->delete_pool(poolname);
}
#endif

#ifdef TEST_PERF_SMALL_PUT_DIRECT
TEST_F(Dawn_client_test, PerfSmallPutDirect)
{
  PMAJOR("Running SmallPutDirect...");
  int rc;

  /* open or create pool */
  Component::IKVStore::pool_t pool =
    _dawn->open_pool(std::string("/mnt/pmem0/dawn") + Options.pool, 0);

  if (pool == Component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = _dawn->create_pool(std::string("/mnt/pmem0/dawn") + Options.pool, GB(1));
  }

  static constexpr unsigned long ITERATIONS = 1000000;
  static constexpr unsigned long VALUE_SIZE = 32;
  static constexpr unsigned long KEY_SIZE   = 8;

  struct record_t {
    std::string key;
    char        value[VALUE_SIZE];
  };

  size_t    data_size = sizeof(record_t) * ITERATIONS;
  record_t *data      = (record_t *) aligned_alloc(MiB(2), data_size);
  madvise(data, data_size, MADV_HUGEPAGE);

  ASSERT_FALSE(data == nullptr);
  auto handle = _dawn->register_direct_memory(
      data, data_size); /* register whole region */

  /* set up data */
  for (unsigned long i = 0; i < ITERATIONS; i++) {
    auto val    = Common::random_string(VALUE_SIZE);
    data[i].key = Common::random_string(KEY_SIZE);
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned long i = 0; i < ITERATIONS; i++) {
    /* different value each iteration; tests memory region registration */
    rc = _dawn->put_direct(pool, data[i].key, data[i].value, VALUE_SIZE,
                           handle); /* pass handle from memory registration */
    ASSERT_TRUE(rc == S_OK || rc == -6);
  }

  auto end  = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count() /
              1000.0;
  PMAJOR("PerfSmallPutDirect Ops/Sec: %lu",
         static_cast<unsigned long>(ITERATIONS / secs));

  _dawn->unregister_direct_memory(handle);

  _dawn->close_pool(pool);
  _dawn->delete_pool("/mnt/pmem0/dawn", Options.pool);
}
#endif

#ifdef TEST_PERF_LARGE_PUT_DIRECT
TEST_F(Dawn_client_test, PerfLargePutDirect)
{
  PMAJOR("Running LargePutDirect...");

  int rc;
  ASSERT_TRUE(_dawn);
  
  const std::string poolname = Options.pool + "/PerfLargePutDirect";
  
  /* open or create pool */
  auto pool = _dawn->create_pool(poolname, GB(8));

  PLOG("Test pool created OK.");

  static constexpr unsigned long PER_ITERATION = 4;
  static constexpr unsigned long ITERATIONS    = 10;
  static constexpr unsigned long VALUE_SIZE    = MB(512);
  static constexpr unsigned long KEY_SIZE      = 16;

  struct record_t {
    std::string key;
    char        value[VALUE_SIZE];
  };

  PLOG("Allocating buffer with test data ...");
  size_t    data_size = sizeof(record_t) * PER_ITERATION;
  record_t *data      = (record_t *) aligned_alloc(MiB(2), data_size);
  madvise(data, data_size, MADV_HUGEPAGE);

  ASSERT_FALSE(data == nullptr);
  auto handle = _dawn->register_direct_memory(data, data_size); /* register whole region */

  PLOG("Filling data...");
  /* set up data */
  for (unsigned long i = 0; i < PER_ITERATION; i++) {
    auto val    = Common::random_string(VALUE_SIZE);
    data[i].key = Common::random_string(KEY_SIZE);
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }
  PLOG("Starting put operation...");

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned long i = 0; i < (ITERATIONS * PER_ITERATION); i++) {
    /* different value each iteration; tests memory region registration */
    rc = _dawn->put_direct(pool, data[i % PER_ITERATION].key,
                           data[i % PER_ITERATION].value, VALUE_SIZE,
                           handle); /* pass handle from memory registration */
    ASSERT_TRUE(rc == S_OK);
  }

  auto end  = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count() / 1000.0;
  
  PINF("PerfLargePutDirect Throughput: %.2f MiB/sec",
       ((PER_ITERATION * ITERATIONS * VALUE_SIZE) / secs) / (1024.0 * 1024));

  _dawn->close_pool(pool);
  _dawn->delete_pool(poolname);
  _dawn->unregister_direct_memory(handle);
  PLOG("Unregistered memory");
}
#endif

#ifdef TEST_PERF_LARGE_GET_DIRECT
TEST_F(Dawn_client_test, PerfLargeGetDirect)
{
  int rc;

  PMAJOR("Running LargeGetDirect...");
  const std::string poolname = Options.pool + "/PerfLargeGetDirect";
  auto pool = _dawn->create_pool(poolname, GB(8));
  ASSERT_TRUE(pool != IKVStore::POOL_ERROR);

  static constexpr unsigned long PER_ITERATION = 4;
  static constexpr unsigned long ITERATIONS    = 10;
  static constexpr unsigned long VALUE_SIZE    = MB(64);
  static constexpr unsigned long KEY_SIZE      = 16;

  struct record_t {
    std::string key;
    char        value[VALUE_SIZE];
  };

  PLOG("Allocating buffer with test data ...");
  size_t    data_size = sizeof(record_t) * PER_ITERATION;
  record_t *data      = (record_t *) aligned_alloc(MiB(2), data_size);
  madvise(data, data_size, MADV_HUGEPAGE);

  ASSERT_FALSE(data == nullptr);
  auto handle = _dawn->register_direct_memory(data, data_size); /* register whole region */

  PLOG("LargeGetDirect: Filling data...");
  /* set up data */
  for (unsigned long i = 0; i < PER_ITERATION; i++) {
    auto val    = Common::random_string(VALUE_SIZE);
    data[i].key = Common::random_string(KEY_SIZE);
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }
  PLOG("Starting PUT operation...");

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned long i = 0; i < (ITERATIONS * PER_ITERATION); i++) {
    PLOG("Putting key(%s)", data[i % PER_ITERATION].key.c_str());
    /* different value each iteration; tests memory region registration */
    rc = _dawn->put_direct(pool,
                           data[i % PER_ITERATION].key,
                           data[i % PER_ITERATION].value,
                           VALUE_SIZE,
                           handle); /* pass handle from memory registration */
    ASSERT_TRUE(rc == S_OK || rc == -6);
  }

  auto end  = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PINF("LargePutDirect Throughput: %.2f MiB/sec",
       ((PER_ITERATION * ITERATIONS * VALUE_SIZE) / secs) / (1024.0 * 1024));

  PINF("LargeGetDirect: Starting .. get phase");

  start = std::chrono::high_resolution_clock::now();

  /* perform get phase */
  for (unsigned long i = 0; i < (ITERATIONS * PER_ITERATION); i++) {
    size_t value_len = VALUE_SIZE;
    PLOG("get_direct (key=%s)", data[i % PER_ITERATION].key.c_str());
    rc               = _dawn->get_direct(pool,
                                         data[i % PER_ITERATION].key,
                                         data[i % PER_ITERATION].value,
                                         value_len,
                                         handle);
    ASSERT_TRUE(rc == S_OK);  // || rc == -6);
  }

  end  = std::chrono::high_resolution_clock::now();
  secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PINF("PerfLargeGetDirect Throughput: %.2f MiB/sec",
       ((PER_ITERATION * ITERATIONS * VALUE_SIZE) / secs) / (1024.0 * 1024));

  _dawn->close_pool(pool);
  _dawn->delete_pool(poolname);
  _dawn->unregister_direct_memory(handle);
}
#endif

#ifdef TEST_PERF_SMALL_GET_DIRECT
TEST_F(Dawn_client_test, PerfSmallGetDirect)
{
  PMAJOR("Running SmallGetDirect...");
  int rc;

  const std::string poolname = Options.pool + "/PerfSmallGetDirect";
  auto pool = _dawn->create_pool(poolname, GB(8));

  static constexpr unsigned long PER_ITERATION = 8;
  static constexpr unsigned long ITERATIONS    = 100000;
  static constexpr unsigned long VALUE_SIZE    = 32;
  static constexpr unsigned long KEY_SIZE      = 8;

  struct record_t {
    std::string key;
    char        value[VALUE_SIZE];
  };

  PLOG("Allocating buffer with test data ...");
  size_t    data_size = sizeof(record_t) * PER_ITERATION;
  record_t *data      = (record_t *) aligned_alloc(MiB(2), data_size);
  madvise(data, data_size, MADV_HUGEPAGE);

  ASSERT_FALSE(data == nullptr);
  auto handle = _dawn->register_direct_memory(
      data, data_size); /* register whole region */

  PLOG("Filling data...");
  /* set up data */
  for (unsigned long i = 0; i < PER_ITERATION; i++) {
    auto val    = Common::random_string(VALUE_SIZE);
    data[i].key = Common::random_string(KEY_SIZE);
    memcpy(data[i].value, val.c_str(), VALUE_SIZE);
  }
  PLOG("Starting PUT operation...");

  auto start = std::chrono::high_resolution_clock::now();

  for (unsigned long i = 0; i < (ITERATIONS * PER_ITERATION); i++) {
    /* different value each iteration; tests memory region registration */
    rc = _dawn->put_direct(pool, data[i % PER_ITERATION].key,
                           data[i % PER_ITERATION].value, VALUE_SIZE,
                           handle); /* pass handle from memory registration */
    ASSERT_TRUE(rc == S_OK || rc == -2);
  }

  auto end  = std::chrono::high_resolution_clock::now();
  auto secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count() /
              1000.0;
  PINF("PerfSmallGet Prep-Throughput: %.2f MiB/sec",
       ((PER_ITERATION * ITERATIONS * VALUE_SIZE) / secs) / (1024.0 * 1024));

  PINF("Starting .. get phase");

  start = std::chrono::high_resolution_clock::now();

  /* perform get phase */
  for (unsigned long i = 0; i < (ITERATIONS * PER_ITERATION); i++) {
    size_t value_len = VALUE_SIZE;
    rc               = _dawn->get_direct(pool, data[i % PER_ITERATION].key,
                           data[i % PER_ITERATION].value, value_len, handle);
    ASSERT_TRUE(rc == S_OK);  // || rc == -6);
  }

  end  = std::chrono::high_resolution_clock::now();
  secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
             .count() /
         1000.0;
  PINF("PerfSmallGet Throughput: %.2f MiB/sec",
       ((PER_ITERATION * ITERATIONS * VALUE_SIZE) / secs) / (1024.0 * 1024));

  _dawn->close_pool(pool);
  _dawn->delete_pool(poolname);
  _dawn->unregister_direct_memory(handle);
}
#endif


TEST_F(Dawn_client_test, Release)
{
  PLOG("Releasing instance...");

  /* release instance */
  _dawn->release_ref();
}

}  // namespace

int main(int argc, char **argv)
{
  //#  option_addr = (argc > 1) ? argv[1] : "10.0.0.41:11911";

  try {
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()("help", "Show help")(
        "debug", po::value<unsigned>()->default_value(0), "Debug level 0-3")(
        "server-addr",
        po::value<std::string>()->default_value("10.0.0.21:11911"),
        "Server address IP:PORT")(
        "device", po::value<std::string>()->default_value("mlx5_0"),
        "Network device (e.g., mlx5_0)")(
        "pool", po::value<std::string>()->default_value("myPool"), "Pool name")(
        "base", po::value<unsigned>()->default_value(0), "Base core.");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") > 0) {
      std::cout << desc;
      return -1;
    }

    Options.addr        = vm["server-addr"].as<std::string>();
    Options.debug_level = vm["debug"].as<unsigned>();
    Options.pool        = vm["pool"].as<std::string>();
    Options.device      = vm["device"].as<std::string>();
    Options.base_core   = vm["base"].as<unsigned>();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
  }
  catch (...) {
    PLOG("bad command line option configuration");
    return -1;
  }

  return 0;
}
