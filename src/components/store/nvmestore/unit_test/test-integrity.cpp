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

#include <stdlib.h>
#include <unordered_map>

#include <gperftools/profiler.h>
#include <boost/crc.hpp>

#define PMEM_PATH "/mnt/pmem0/pool-nvmestore"
#define POOL_NAME "test-integrity.pool"

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
  static Component::IKVStore::pool_t _pool;
};

Component::IKVStore *       KVStore_test::_kvstore;
Component::IKVStore::pool_t KVStore_test::_pool;

static constexpr size_t KEY_LEN = 16;
#ifdef BIG_ELEM
static constexpr int    nr_elem = 4;  // number of the test elem in the pool
static constexpr size_t VAL_LEN = GB(1);
#else
// static constexpr int nr_elem = 10000;  // TODO: Takes long if I dont cache
// key list
static constexpr int    nr_elem = 100;  // number of the test elem in the pool
static constexpr size_t VAL_LEN = KB(128);
#endif
std::unordered_map<std::string, int> _crc_map;

TEST_F(KVStore_test, Instantiate)
{
  /* create object instance through factory */
  Component::IBase *comp = Component::load_component(
      "libcomanche-nvmestore.so", Component::nvmestore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory *fact =
      (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  std::map<std::string, std::string> params;
  params["owner"]      = "testowner";
  params["name"]       = "testname";
  params["pci"]        = opt.pci;
  params["pm_path"]    = "/mnt/pmem0/";
  unsigned debug_level = 0;

  // this nvme-store use a block device and a block allocator
  _kvstore = fact->create(debug_level, params);

  fact->release_ref();
}

TEST_F(KVStore_test, OpenPool)
{
  PLOG(" test-nvmestore: try to openpool");
  ASSERT_TRUE(_kvstore);
  // pass blk and alloc here
  _pool = _kvstore->create_pool(PMEM_PATH POOL_NAME, GB(8));
  ASSERT_TRUE(_pool > 0);
}

TEST_F(KVStore_test, BasicPut)
{
  ASSERT_TRUE(_pool);
  PINF("Writing elems...");
  for (int i = 0; i < nr_elem; i++) {
    std::string key = "MyKey" + std::to_string(i);
    key.resize(KEY_LEN, '.');
    auto val = Common::random_string(VAL_LEN);

    boost::crc_32_type result;
    result.process_bytes(val.data(), val.length());
    PDBG("Checksum: %d", result.checksum());

    _crc_map[key] = result.checksum();

    EXPECT_TRUE(S_OK == _kvstore->put(_pool, key, val.c_str(), VAL_LEN));
  }
}

TEST_F(KVStore_test, GetDirect)
{
  IKVStore::memory_handle_t handle;
  void *                    value     = nullptr;
  size_t                    value_len = 0;

  ASSERT_EQ(S_OK, _kvstore->allocate_direct_memory(value, VAL_LEN, handle));

  for (int i = 0; i < nr_elem; i++) {
    std::string key = "MyKey" + std::to_string(i);
    key.resize(KEY_LEN, '.');

    _kvstore->get_direct(_pool, key, value, value_len, handle);

    boost::crc_32_type result;
    result.process_bytes(value, value_len);
    PDBG("Checksum: %d", result.checksum());
    EXPECT_EQ(_crc_map[key], result.checksum());

    PDBG("Value=(%.50s) %lu", ((char *) value), value_len);
  }

  _kvstore->free_direct_memory(handle);
}

TEST_F(KVStore_test, BasicGet)
{
  for (int i = 0; i < nr_elem; i++) {
    std::string key = "MyKey" + std::to_string(i);
    key.resize(KEY_LEN, '.');

    void * value     = nullptr;
    size_t value_len = 0;
    _kvstore->get(_pool, key, value, value_len);

    boost::crc_32_type result;
    result.process_bytes(value, value_len);
    PDBG("Checksum: %d", result.checksum());
    EXPECT_EQ(_crc_map[key], result.checksum());

    free(value);

    // PINF("Value=(%.50s) %lu", ((char*)value), value_len);
  }
}

TEST_F(KVStore_test, BasicMap)
{
  auto value_len_sum = 0;

  _kvstore->map(_pool,
                [&value_len_sum](const std::string &key, const void *value,
                                 const size_t value_len) -> int {
                  value_len_sum += value_len;
                  return 0;
                });
  EXPECT_EQ(nr_elem * VAL_LEN, value_len_sum);
}

TEST_F(KVStore_test, BasicMapKeys)
{
  auto   key_len_sum = 0;
  size_t key_len     = KEY_LEN;
  _kvstore->map_keys(_pool, [&key_len_sum](const std::string &key) -> int {
    key_len_sum += key.size();
    return 0;
  });
  EXPECT_EQ(nr_elem * KEY_LEN, key_len_sum);
}

TEST_F(KVStore_test, BasicErase)
{
  ASSERT_EQ(nr_elem, _kvstore->count(_pool));
  for (int i = 0; i < nr_elem; i++) {
    std::string key = "MyKey" + std::to_string(i);
    key.resize(KEY_LEN, '.');
    _kvstore->erase(_pool, key);
  }

  ASSERT_EQ(0, _kvstore->count(_pool));
}

TEST_F(KVStore_test, ClosePool) { _kvstore->close_pool(_pool); }

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
