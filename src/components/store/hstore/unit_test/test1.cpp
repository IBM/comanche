#include "store_map.h"

#include <gtest/gtest.h>
#include <common/utils.h>
#include <api/components.h>
/* note: we do not include component source, only the API definition */
#include <api/kvstore_itf.h>

#include <algorithm>
#include <random>
#include <sstream>
#include <string>

using namespace Component;

#define USE_PMEMOBJ 1

namespace {

// The fixture for testing class Foo.
class KVStore_test : public ::testing::Test {

  static constexpr std::size_t many_count_target_large = 2000000;
  /* Shorter test: use when PMEM_IS_PMEM_FORCE=0 */
  static constexpr std::size_t many_count_target_small = 400;

  static constexpr std::size_t estimated_object_count_large = many_count_target_large;
  /* More testing of table splits, at a performance cost */
  static constexpr std::size_t estimated_object_count_small = 1;

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
  /* persistent memory if enabled at all, is simulated and not real */
  static bool pmem_simulated;
  /* persistent memory is effective (either real, indicated by no PMEM_IS_PMEM_FORCE or simulated by PMEM_IS_PMEM_FORCE 0 not 1 */
  static bool pmem_effective;
  static Component::IKVStore * _kvstore;
  static Component::IKVStore::pool_t pool;

  static const std::size_t estimated_object_count;

  static std::string single_key;
  static std::string single_value;
  static std::size_t single_value_size;
  static std::string single_value_updated_same_size;
  static std::string single_value_updated_different_size;
  static std::string single_value_updated3;
  static std::size_t single_count;

  static constexpr unsigned many_key_length = 8;
  static constexpr unsigned many_value_length = 16;
  using kv_t = std::tuple<std::string, std::string>;
  static std::vector<kv_t> kvv;
  static const std::size_t many_count_target;
  static std::size_t many_count_actual;

  /* NOTE: ignoring the remote possibility of a random number collision in the first lock_count entries */
  static const std::size_t lock_count;

  static std::size_t extant_count; /* Number of PutMany keys not placed because they already existed */

  std::string pool_dir() const
  {
#if USE_PMEMOBJ
    return "/mnt/pmem0/pool/0/";
#else
    return "/dev";
#endif
  }

  std::string pool_name()
  {
#if USE_PMEMOBJ
    return "test-" + store_map::impl->name + ".pool";
#else
    return "dax0.0";
#endif
  }
};

constexpr std::size_t KVStore_test::estimated_object_count_small;
constexpr std::size_t KVStore_test::estimated_object_count_large;
constexpr std::size_t KVStore_test::many_count_target_small;
constexpr std::size_t KVStore_test::many_count_target_large;

bool KVStore_test::pmem_simulated = getenv("PMEM_IS_PMEM_FORCE");
bool KVStore_test::pmem_effective = ! getenv("PMEM_IS_PMEM_FORCE") || getenv("PMEM_IS_PMEM_FORCE") == std::string("0");
Component::IKVStore * KVStore_test::_kvstore;
Component::IKVStore::pool_t KVStore_test::pool;

const std::size_t KVStore_test::estimated_object_count = pmem_simulated ? estimated_object_count_small : estimated_object_count_large;

/* Keys 23-byte or fewer are stored inline. Provide one longer to force allocation */
std::string KVStore_test::single_key = "MySingleKeyLongEnoughToForceAllocation";
std::string KVStore_test::single_value         = "Hello world!";
std::size_t KVStore_test::single_value_size    = MB(8);
std::string KVStore_test::single_value_updated_same_size = "Jello world!";
std::string KVStore_test::single_value_updated_different_size = "Hello world!";
std::string KVStore_test::single_value_updated3 = "WeXYZ world!";
std::size_t KVStore_test::single_count = 1U;

constexpr unsigned KVStore_test::many_key_length;
constexpr unsigned KVStore_test::many_value_length;
const std::size_t KVStore_test::many_count_target = pmem_simulated ? many_count_target_small : many_count_target_large;
std::size_t KVStore_test::many_count_actual;
std::size_t KVStore_test::extant_count = 0;
std::vector<KVStore_test::kv_t> KVStore_test::kvv;

const std::size_t KVStore_test::lock_count = 60;

TEST_F(KVStore_test, Instantiate)
{
  /* create object instance through factory */
  auto link_library = "libcomanche-" + store_map::impl->name + ".so";
  Component::IBase * comp = Component::load_component(link_library,
                                                      store_map::impl->factory_id);

  ASSERT_TRUE(comp);
  auto fact = static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid()));

  _kvstore = fact->create("owner", "name", store_map::location);

  fact->release_ref();
}

TEST_F(KVStore_test, RemoveOldPool)
{
  if ( _kvstore )
  {
    try
    {
      pool = _kvstore->open_pool(pool_dir(), pool_name(), 0);
      if ( 0 < int64_t(pool) )
      {
        _kvstore->delete_pool(pool);
      }
    }
    catch ( Exception & )
    {
    }
  }
}

TEST_F(KVStore_test, CreatePool)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->create_pool(pool_dir(), pool_name(), many_count_target * 4U * 64U + 4 * single_value_size, 0, estimated_object_count);
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
  single_value.resize(single_value_size);

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

/* hstore issue 41 specifies different implementations for same-size replace vs different-size replace. */
TEST_F(KVStore_test, BasicReplaceSameSize)
{
  {
    single_value_updated_same_size.resize(single_value_size);
    auto r = _kvstore->put(pool, single_key, single_value_updated_same_size.data(), single_value_updated_same_size.length());
    EXPECT_EQ(r, S_OK);
  }
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(r, S_OK);
  PINF("Value=(%.50s) %lu", static_cast<char *>(value), value_len);
  EXPECT_EQ(0, memcmp(single_value_updated_same_size.data(), value, single_value_updated_same_size.size()));
  _kvstore->free_memory(value);
}

TEST_F(KVStore_test, BasicReplaceDifferentSize)
{
  {
    auto r = _kvstore->put(pool, single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length());
    EXPECT_EQ(r, S_OK);
  }
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(r, S_OK);
  PINF("Value=(%.50s) %lu", static_cast<char *>(value), value_len);
  EXPECT_EQ(0, memcmp(single_value_updated_different_size.data(), value, single_value_updated_different_size.size()));
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
    void * old_value = nullptr;
    size_t old_value_len = 0;
    if ( S_OK == _kvstore->get(pool, key, old_value, old_value_len) )
    {
      _kvstore->free_memory(old_value);
      ++extant_count;
    }
    else
    {
      auto r = _kvstore->put(pool, key, value.data(), value.length());
      EXPECT_EQ(r, S_OK);
      if ( r == S_OK )
      {
        ++many_count_actual;
      }
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

TEST_F(KVStore_test, Count1)
{
  auto count = _kvstore->count(pool);
  /* count should reflect Put, PutMany */
  EXPECT_EQ(count, single_count + many_count_actual);
}

TEST_F(KVStore_test, CountByBucket)
{
  std::uint64_t count = 0;
  _kvstore->debug(pool, 2 /* COUNT_BY_BUCKET */, reinterpret_cast<std::uint64_t>(&count));
  /* should reflect Put, PutMany */
  EXPECT_EQ(count, single_count + many_count_actual);
}

TEST_F(KVStore_test, ClosePool)
{
  if ( pmem_effective )
  {
    _kvstore->close_pool(pool);
  }
}

TEST_F(KVStore_test, OpenPool)
{
  ASSERT_TRUE(_kvstore);
  if ( pmem_effective )
  {
    pool = _kvstore->open_pool(pool_dir(), pool_name(), 0);
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
    std::size_t mismatch_count = 0;
    for ( auto &kv : kvv )
    {
      const auto &key = std::get<0>(kv);
      const auto &ev = std::get<1>(kv);
      void * value = nullptr;
      size_t value_len = 0;
      auto r = _kvstore->get(pool, key, value, value_len);
      EXPECT_EQ(r, S_OK);
      EXPECT_EQ(value_len, many_value_length);
      mismatch_count += ( 0 != memcmp(ev.data(), value, ev.size()) );
      _kvstore->free_memory(value);
    }
    EXPECT_EQ(mismatch_count, extant_count);
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
    std::vector<std::unique_ptr<IKVStore::Operation>> v;
    v.emplace_back(std::make_unique<IKVStore::Operation_write>(0, 1, "W"));
    v.emplace_back(std::make_unique<IKVStore::Operation_write>(2, 3, "XYZ"));
    std::vector<IKVStore::Operation *> v2;
    std::transform(v.begin(), v.end(), std::back_inserter(v2), [] (const auto &i) { return i.get(); });
    auto r =
      _kvstore->atomic_update(
      pool
      , single_key
      , v2
    );
    EXPECT_EQ(r, S_OK);
  }

  {
    void * value = nullptr;
    size_t value_len = 0;
    auto r = _kvstore->get(pool, single_key, value, value_len);
    EXPECT_EQ(r, S_OK);
    PINF("Value=(%.50s) %lu", static_cast<char *>(value), value_len);
    EXPECT_EQ(value_len, single_value_updated_different_size.size());
    EXPECT_EQ(0, memcmp(single_value_updated3.data(), value, single_value_updated3.size()));
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
