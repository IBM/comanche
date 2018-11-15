#include "store_map.h"

#include <gtest/gtest.h>
#include <common/utils.h>
#include <api/components.h>
/* note: we do not include component source, only the API definition */
#include <api/kvstore_itf.h>

#include <string>
#include <random>
#include <sstream>

using namespace Component;

namespace {

// The fixture for testing class Foo.
class KVStore_test : public ::testing::Test {

  static constexpr unsigned estimated_object_count_large = 5000;
  static constexpr unsigned many_count_target_large = 20000;
  /* More testing of table splits, at a performance cost */
  static constexpr unsigned estimated_object_count_small = 1;
  /* Shorter test: use when PMEM_IS_PMEM_FORCE=0 */
  static constexpr unsigned many_count_target_small = 400;

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
  static bool pmem_force;
  static Component::IKVStore * _kvstore;
  static Component::IKVStore::pool_t pool;

  static const unsigned estimated_object_count;

  static std::string single_key;
  static std::string single_value;
  static std::string single_value_updated;
  static std::size_t single_count;

  static constexpr unsigned many_key_length = 8;
  static constexpr unsigned many_value_length = 16;
  using kv_t = std::tuple<std::string, std::string>;
  static std::vector<kv_t> kvv;
  static const unsigned many_count_target;
  static unsigned many_count_actual;

  /* NOTE: ingoring the remote possibility of a random number collision in the first lock_count entries */
  static const unsigned lock_count;
};

constexpr unsigned KVStore_test::estimated_object_count_small;
constexpr unsigned KVStore_test::estimated_object_count_large;
constexpr unsigned KVStore_test::many_count_target_small;
constexpr unsigned KVStore_test::many_count_target_large;

bool KVStore_test::pmem_force = getenv("PMEM_IS_PMEM_FORCE") && getenv("PMEM_IS_PMEM_FORCE") == std::string("1");
Component::IKVStore * KVStore_test::_kvstore;
Component::IKVStore::pool_t KVStore_test::pool;

const unsigned KVStore_test::estimated_object_count = KVStore_test::pmem_force ? estimated_object_count_large : estimated_object_count_small;

std::string KVStore_test::single_key = "MySingleKeyLongEnoughToFoceAllocation";
std::string KVStore_test::single_value         = "Hello world!";
std::string KVStore_test::single_value_updated = "XeXXX world!";
std::size_t KVStore_test::single_count = 1U;

constexpr unsigned KVStore_test::many_key_length;
constexpr unsigned KVStore_test::many_value_length;
const unsigned KVStore_test::many_count_target = KVStore_test::pmem_force ? many_count_target_large : many_count_target_small;
unsigned KVStore_test::many_count_actual = 0;
std::vector<KVStore_test::kv_t> KVStore_test::kvv;

const unsigned KVStore_test::lock_count = 60;

#define PMEM_PATH "/mnt/pmem0/pool/0/"
//#define PMEM_PATH "/dev/pmem0"


TEST_F(KVStore_test, Instantiate)
{
  /* create object instance through factory */
  auto link_library = "libcomanche-" + store_map::impl->name + ".so";
  Component::IBase * comp = Component::load_component(link_library,
                                                      store_map::impl->factory_id);

  ASSERT_TRUE(comp);
  auto fact = static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid()));

  _kvstore = fact->create("owner","name");

  fact->release_ref();
}

TEST_F(KVStore_test, RemoveOldPool)
{
  if ( _kvstore )
  {
    try
    {
      pool = _kvstore->open_pool(PMEM_PATH, "test-" + store_map::impl->name + ".pool", MB(128UL));
      if ( 0 < int64_t(pool) )
      {
        _kvstore->delete_pool(pool);
      }
    }
    catch ( General_exception & )
    {
    }
  }
}

TEST_F(KVStore_test, CreatePool)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->create_pool(PMEM_PATH, "test-" + store_map::impl->name + ".pool", MB(128UL), 0, estimated_object_count);
  ASSERT_LT(0, int64_t(pool));
}

TEST_F(KVStore_test, BasicGet0)
{
  void * value = nullptr;
  size_t value_len = 0;

  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_NE(r, S_OK);
  if( r == S_OK )
  {
    ASSERT_EQ("Key already exists", "Did you forget to delete the pool before running the test?");
  }
  _kvstore->free_memory(value);
}

TEST_F(KVStore_test, BasicPut)
{
  single_value.resize(MB(8));

  auto r = _kvstore->put(pool, single_key, single_value.data(), single_value.length());
  EXPECT_EQ(r, S_OK);
}

TEST_F(KVStore_test, BasicGet1)
{
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(r, S_OK);
  PINF("Value=(%.50s) %lu", static_cast<char *>(value), value_len);
  EXPECT_EQ(0, memcmp(single_value.data(), value, single_value.size()));
  _kvstore->free_memory(value);
}

TEST_F(KVStore_test, PopulateMany)
{
  std::mt19937_64 r0{};
  for ( auto i = 0; i != many_count_target; ++i )
  {
    auto ukey = r0();
    std::ostringstream s;
    s << std::hex << ukey;
    auto key = s.str();
    key.resize(many_key_length, '.');
    auto value = std::to_string(i);
    value.resize(many_value_length, '.');
    kvv.emplace_back(key, value);
  }
}

TEST_F(KVStore_test, PutMany)
{
  many_count_actual = 0;

  for ( auto &kv : kvv )
  {
    const auto &key = std::get<0>(kv);
    const auto &value = std::get<1>(kv);
#if 0
    auto key = s.str();
    key.resize(many_key_length, '.');
    auto value = std::to_string(i);
    value.resize(many_value_length, '.');
#endif
    auto r = _kvstore->put(pool, key, value.data(), value.length());
    if ( r == S_OK )
    {
      ++many_count_actual;
    }
    else
    {
      std::cerr << __func__ << " FAIL " << key << "\n";
    }
  }
  EXPECT_LE(many_count_actual, many_count_target);
  EXPECT_LE(many_count_target * 0.99, double(many_count_actual));
}

TEST_F(KVStore_test, BasicMap)
{
  _kvstore->map(pool,[](const std::string &key,
                        const void * value,
                        const size_t value_len) -> int
                {
#if 0
                    PINF("key:%lx value@%p value_len=%lu", key, value, value_len);
#endif
                    return 0;
                  });
}

TEST_F(KVStore_test, Size1)
{
  auto count = _kvstore->count(pool);
  /* count should reflect Put, PutMany */
  EXPECT_EQ(count, single_count + many_count_actual);
}

TEST_F(KVStore_test, ClosePool)
{
  if ( ! pmem_force )
  {
    _kvstore->close_pool(pool);
  }
}

TEST_F(KVStore_test, OpenPool)
{
  ASSERT_TRUE(_kvstore);
  if ( ! pmem_force )
  {
    pool = _kvstore->open_pool(PMEM_PATH, "test-hstore.pool", MB(128));
  }
  ASSERT_LT(0, int64_t(pool));
}

TEST_F(KVStore_test, Size2a)
{
  auto count = _kvstore->count(pool);
  /* count should reflect Put, PutMany */
  EXPECT_EQ(count, single_count + many_count_actual);
}

TEST_F(KVStore_test, BasicGet2)
{
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(r, S_OK);
  PINF("Value=(%.50s) %lu", static_cast<char *>(value), value_len);
  _kvstore->free_memory(value);
}

TEST_F(KVStore_test, Size2b)
{
  auto count = _kvstore->count(pool);
  /* count should reflect PutMany */
  EXPECT_EQ(count, single_count + many_count_actual);
}

TEST_F(KVStore_test, GetMany)
{
  for ( auto i = 0; i != 10; ++i )
  {
    for ( auto &kv : kvv )
    {
      const auto &key = std::get<0>(kv);
      const auto &ev = std::get<1>(kv);
      void * value = nullptr;
      size_t value_len = 0;
      auto r = _kvstore->get(pool, key, value, value_len);
      EXPECT_EQ(r, S_OK);
      EXPECT_EQ(value_len, many_value_length);
      EXPECT_EQ(0, memcmp(ev.data(), value, ev.size()));
      _kvstore->free_memory(value);
    }
  }
}

TEST_F(KVStore_test, LockMany)
{
  unsigned ct = 0;
  for ( auto &kv : kvv )
  {
    if ( ct == lock_count ) { break; }
    const auto &key = std::get<0>(kv);
    const auto &ev = std::get<1>(kv);
    const auto key_new = std::get<0>(kv) + "x";
    void *value0 = nullptr;
    std::size_t value0_len = 0;
    auto r0 = _kvstore->lock(pool, key, IKVStore::STORE_LOCK_READ, value0, value0_len);
    EXPECT_NE(r0, nullptr);
    EXPECT_EQ(value0_len, many_value_length);
    EXPECT_EQ(0, memcmp(ev.data(), value0, ev.size()));
    void * value1 = nullptr;
    std::size_t value1_len = 0;
    auto r1 = _kvstore->lock(pool, key, IKVStore::STORE_LOCK_READ, value1, value1_len);
    EXPECT_NE(r1, nullptr);
    EXPECT_EQ(value1_len, many_value_length);
    EXPECT_EQ(0, memcmp(ev.data(), value1, ev.size()));
    /* Exclusive locking test. Skip if the library is built without locking. */
    if ( _kvstore->thread_safety() == IKVStore::THREAD_MODEL_MULTI_PER_POOL )
    {
      void * value2 = nullptr;
      std::size_t value2_len = 0;
      auto r2 = _kvstore->lock(pool, key, IKVStore::STORE_LOCK_WRITE, value2, value2_len);
      EXPECT_EQ(r2, nullptr);
    }
    void * value3 = nullptr;
    std::size_t value3_len = many_value_length;
    auto r3 = _kvstore->lock(pool, key_new, IKVStore::STORE_LOCK_WRITE, value3, value3_len);
    EXPECT_NE(r3, nullptr);
    EXPECT_EQ(value3_len, many_value_length);
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

TEST_F(KVStore_test, Size2c)
{
  auto count = _kvstore->count(pool);
  /* count should reflect Put, PutMany and LockMany */
  EXPECT_EQ(count, single_count + many_count_actual + lock_count);
}

/* Missing:
 *  - test of invalid parameters
 *    - offsets greater than sizeof data
 *    - non-existent key
 *    - invalid operations
 *  - test of crash recovery
 */
TEST_F(KVStore_test, BasicUpdate)
{
  {
    auto op_write = IKVStore::OP_WRITE;
    auto r = _kvstore->atomic_update(pool, single_key, {{0, 1, op_write}, {2,3,op_write}});
    EXPECT_EQ(r, S_OK);
  }

  {
    void * value = nullptr;
    size_t value_len = 0;
    auto r = _kvstore->get(pool, single_key, value, value_len);
    EXPECT_EQ(r, S_OK);
    PINF("Value=(%.50s) %lu", static_cast<char *>(value), value_len);
    EXPECT_EQ(value_len, single_value.size());
    EXPECT_EQ(0, memcmp(single_value_updated.data(), value, single_value_updated.size()));
    _kvstore->free_memory(value);
  }

  auto count = _kvstore->count(pool);
  EXPECT_EQ(count, single_count + many_count_actual + lock_count);
}

TEST_F(KVStore_test, BasicErase)
{
  {
    auto r = _kvstore->erase(pool, single_key);
    EXPECT_EQ(r, S_OK);
  }

  auto count = _kvstore->count(pool);
  EXPECT_EQ(count, many_count_actual + lock_count);
}

TEST_F(KVStore_test, EraseMany)
{
  auto erase_count = 0;
  for ( auto &kv : kvv )
  {
    const auto &key = std::get<0>(kv);
    auto r = _kvstore->erase(pool, key);
    if ( r == S_OK )
    {
      ++erase_count;
    }
  }
  EXPECT_LE(many_count_actual, erase_count);
  auto count = _kvstore->count(pool);
  EXPECT_EQ(count, lock_count);
}

TEST_F(KVStore_test, Size3)
{
}

TEST_F(KVStore_test, DeletePool)
{
  _kvstore->delete_pool(pool);
}

} // namespace

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
