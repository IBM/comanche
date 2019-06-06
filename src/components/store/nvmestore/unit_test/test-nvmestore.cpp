/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 *
 */

/*
 * Authors:
 *
 * Feng Li (fengggli@yahoo.com)
 *
 */

/* note: we do not include component source, only the API definition */
#include <api/block_allocator_itf.h>
#include <api/block_itf.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <common/str_utils.h>
#include <common/utils.h>
#include <core/physical_memory.h>
#include <gtest/gtest.h>
#include "data.h"

#include <stdlib.h>

#include <gperftools/profiler.h>

#define PMEM_PATH "/mnt/pmem0/pool-nvmestore"
#define POOL_NAME "test-basic.pool"

#define DO_BASIC_TEST

//#define USE_FILESTORE
#undef USE_FILESTORE

using namespace Component;

struct {
  std::string pci;
} opt;

namespace
{
// The fixture for testing class Foo.
class KVStore_test : public ::testing::Test {
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
  static Component::IKVStore *       _kvstore;
  static Component::IKVStore *       _kvstore2;
  static Component::IKVStore::pool_t _pool;
  static std::string                 _pool_fullname;
  static bool _pool_is_reopen;  // this run is open a previously created pool

  using kv_t = std::tuple<std::string, std::string>;
  static std::vector<kv_t>  kvv;
  static constexpr unsigned single_value_length = MB(8);
};

Component::IKVStore *       KVStore_test::_kvstore;
Component::IKVStore *       KVStore_test::_kvstore2;
Component::IKVStore::pool_t KVStore_test::_pool;
std::string                 KVStore_test::_pool_fullname;
bool                        KVStore_test::_pool_is_reopen;

constexpr unsigned              KVStore_test::single_value_length;
std::vector<KVStore_test::kv_t> KVStore_test::kvv;

TEST_F(KVStore_test, Instantiate)
{
  /* create object instance through factory */
#ifndef USE_FILESTORE
  Component::IBase *comp = Component::load_component(
      "libcomanche-nvmestore.so", Component::nvmestore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory *fact =
      (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  // this nvme-store use a block device and a block allocator
  std::map<std::string, std::string> params;
  params["owner"]      = "testowner";
  params["name"]       = "testname";
  params["pci"]        = opt.pci;
  params["pm_path"]    = "/mnt/pmem0/";
  unsigned debug_level = 0;

  _kvstore = fact->create(debug_level, params);
#else
  Component::IBase *comp = Component::load_component(
      "libcomanche-storefile.so", Component::filestore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory *fact =
      (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  // this nvme-store use a block device and a block allocator
  _kvstore = fact->create("owner", "name");
#endif

  fact->release_ref();
}

TEST_F(KVStore_test, OpenPool)
{
  PLOG(" test-nvmestore: try to openpool");
  ASSERT_TRUE(_kvstore);
  // pass blk and alloc here
  std::string pool_path;
  std::string pool_name;

  pool_name = "basic-nr-" + std::to_string(Data::NUM_ELEMENTS) + "-sz-" +
              std::to_string(Data::VAL_LEN) + ".pool";

  pool_path = "./data/";
  try {
    _pool_fullname  = pool_path + pool_name;
    _pool           = _kvstore->create_pool(_pool_fullname, MB(128));
    _pool_is_reopen = false;
  }

  catch (...) {
    // open the pool if it exists
    PINF("NVMEStore:open a exsiting pool instead!");
    _pool           = _kvstore->open_pool(_pool_fullname);
    _pool_is_reopen = true;
  }
  ASSERT_TRUE(_pool > 0);
}

#if 0
TEST_F(KVStore_test, BasicGetEmpty)
{
  std::string key = "MyKey";

  void * value     = nullptr;
  size_t value_len = 0;
  EXPECT_EQ(IKVStore::E_KEY_NOT_FOUND,
            _kvstore->get(_pool, key, value, value_len));

  free(value);
}
#endif

TEST_F(KVStore_test, BasicPut)
{
  ASSERT_TRUE(_pool);
  std::string key   = "MyKey";
  std::string value = "Hello world!";
  //  value.resize(value.length()+1); /* append /0 */
  value.resize(single_value_length);

  kvv.emplace_back(key, value);

  EXPECT_TRUE(S_OK == _kvstore->put(_pool, key, value.c_str(), value.length()));
}

TEST_F(KVStore_test, GetDirect)
{
#if 0
  /*Register Mem is only from gdr memory*/
  //ASSERT_TRUE(S_OK == _kvstore->register_direct_memory(user_buf, MB(8)));
#endif
  IKVStore::memory_handle_t handle;
  std::string               key       = "MyKey";
  void *                    value     = nullptr;
  size_t                    value_len = 0;

  handle = _kvstore->allocate_direct_memory(value, single_value_length);

  _kvstore->get_direct(_pool, key, value, value_len, handle);

  EXPECT_FALSE(strcmp("Hello world!", (char *) value));
  PINF("Value=(%.50s) %lu", ((char *) value), value_len);

  _kvstore->free_direct_memory(handle);
}

TEST_F(KVStore_test, BasicGet)
{
  std::string key = "MyKey";

  void * value     = nullptr;
  size_t value_len = 0;
  _kvstore->get(_pool, key, value, value_len);

  EXPECT_FALSE(strcmp("Hello world!", (char *) value));
  PINF("Value=(%.50s) %lu", ((char *) value), value_len);

  free(value);
}

// put the same key with different value
TEST_F(KVStore_test, PutOverwrite)
{
  ASSERT_TRUE(_pool);
  std::string key   = "MyKey";
  std::string value = "Second Hello world!";
  //  value.resize(value.length()+1); /* append /0 */
  value.resize(single_value_length);

  // also update the record
  for (auto &kv : kvv) {
    // if ( ct == lock_count ) { break; }
    const auto &matched_key = std::get<0>(kv);
    auto &      ev          = std::get<1>(kv);
    if (matched_key == key) ev = value;
  }

  kvv.emplace_back(key, value);

  EXPECT_EQ(S_OK, _kvstore->put(_pool, key, value.c_str(), value.length()));

  void * get_value     = nullptr;
  size_t get_value_len = 0;
  EXPECT_EQ(S_OK, _kvstore->get(_pool, key, get_value, get_value_len));

  EXPECT_FALSE(strcmp("Second Hello world!", (char *) get_value));
  PINF("Value=(%.50s) %lu", ((char *) get_value), get_value_len);

  free(get_value);
}

#if 0
TEST_F(KVStore_test, BasicMap)
{
  _kvstore->map(_pool,
                [](const std::string &key, const void *value,
                   const size_t value_len) -> int {
                  PINF("key:%s, value@%p value_len=%lu", key.c_str(), value,
                       value_len);
                  return 0;
                  ;
                });
}
#endif

TEST_F(KVStore_test, BasicMapKeys)
{
  _kvstore->map_keys(_pool, [](const std::string &key) -> int {
    PINF("key:%s", key.c_str());
    return 0;
    ;
  });
}

/* lock */
TEST_F(KVStore_test, LockBasic)
{
  unsigned ct = 0;

  Component::IKVStore::pool_t pool = _pool;

  for (auto &kv : kvv) {
    // if ( ct == lock_count ) { break; }
    const auto &key        = std::get<0>(kv);
    const auto &ev         = std::get<1>(kv);
    const auto  key_new    = std::get<0>(kv) + "x";
    void *      value0     = nullptr;
    std::size_t value0_len = 0;
    auto r0 = _kvstore->lock(pool, key, IKVStore::STORE_LOCK_READ, value0,
                             value0_len);
    EXPECT_NE(r0, nullptr);
    EXPECT_EQ(value0_len, single_value_length);
    EXPECT_EQ(0, memcmp(ev.data(), value0, ev.size()));
    void *      value1     = nullptr;
    std::size_t value1_len = 0;
    auto r1 = _kvstore->lock(pool, key, IKVStore::STORE_LOCK_READ, value1,
                             value1_len);
    EXPECT_NE(r1, nullptr);
    EXPECT_EQ(value1_len, single_value_length);
    EXPECT_EQ(0, memcmp(ev.data(), value1, ev.size()));
    /* Exclusive locking test. Skip if the library is built without locking. */
    if (_kvstore->thread_safety() == IKVStore::THREAD_MODEL_MULTI_PER_POOL) {
      void *      value2     = nullptr;
      std::size_t value2_len = 0;
      auto r2 = _kvstore->lock(pool, key, IKVStore::STORE_LOCK_WRITE, value2,
                               value2_len);
      EXPECT_EQ(r2, nullptr);
    }
    void *      value3     = nullptr;
    std::size_t value3_len = single_value_length;
    auto r3 = _kvstore->lock(pool, key_new, IKVStore::STORE_LOCK_WRITE, value3,
                             value3_len);
    EXPECT_NE(r3, nullptr);
    EXPECT_EQ(value3_len, single_value_length);
    EXPECT_NE(value3, nullptr);

    auto r0x = _kvstore->unlock(pool, r0);
    EXPECT_EQ(r0x, S_OK);
    auto r1x = _kvstore->unlock(pool, r1);
    EXPECT_EQ(r1x, S_OK);
    auto r3x = _kvstore->unlock(pool, r3);
    EXPECT_EQ(r3x, S_OK);

    ++ct;
  }
}

TEST_F(KVStore_test, BasicErase) { _kvstore->erase(_pool, "MyKey"); }

#if 0
/*
 * multiple store on same nvmedevice will use the same _block and the _blk_alloc
 */
TEST_F(KVStore_test, Multiplestore)
{
  /* create object instance through factory */
  Component::IBase *comp = Component::load_component(
      "libcomanche-nvmestore.so", Component::nvmestore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory *fact =
      (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  std::map<std::string, std::string> params;
  params["owner"]      = "testowner";
  params["name"]       = "anothername";
  params["pci"]        = opt.pci;
  params["pm_path"]    = "/mnt/pmem0/";
  unsigned debug_level = 0;

  // this nvme-store use a block device and a block allocator
  _kvstore2 = fact->create(debug_level, params);

  fact->release_ref();
  Component::IKVStore::pool_t store2_pool2 =
      _kvstore2->create_pool("data/test-nvme3.pool", MB(128));
}

#endif
TEST_F(KVStore_test, ClosePool) { _kvstore->close_pool(_pool); }

TEST_F(KVStore_test, DeletePool) { _kvstore->delete_pool(_pool_fullname); }

TEST_F(KVStore_test, ReleaseStore) { _kvstore->release_ref(); }

}  // namespace

int main(int argc, char **argv)
{
  if (argc != 2) {
    PINF("test-nvmestore <pci-address>");
    return 0;
  }

  opt.pci = argv[1];

  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
