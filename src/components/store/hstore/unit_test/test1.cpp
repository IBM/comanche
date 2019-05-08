/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "store_map.h"

#include <gtest/gtest.h>
#if 0
#include <common/utils.h>
#endif
#include <api/components.h>
/* note: we do not include component source, only the API definition */
#include <api/kvstore_itf.h>

#include <algorithm>
#include <random>
#include <sstream>
#include <string>

using namespace Component;

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
  static std::string missing_key;
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
  static constexpr unsigned get_expand = 2;

  /* NOTE: ignoring the remote possibility of a random number collision in the first lock_count entries */
  static const std::size_t lock_count;

  static std::size_t extant_count; /* Number of PutMany keys not placed because they already existed */

  std::string pool_name() const
  {
    return "/mnt/pmem0/pool/0/test-" + store_map::impl->name + store_map::numa_zone() + ".pool";
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
std::string KVStore_test::missing_key = "KeyNeverInserted";
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
  /* numa node 0 */
  _kvstore = fact->create("owner", "numa0", store_map::location);

  fact->release_ref();
}

TEST_F(KVStore_test, RemoveOldPool)
{
  if ( _kvstore )
  {
    try
    {
      _kvstore->delete_pool(pool_name());
    }
    catch ( Exception & )
    {
    }
  }
}

TEST_F(KVStore_test, CreatePool)
{
  ASSERT_TRUE(_kvstore);
  /* count of elements multiplied
   *  - by 64 for bucket size,
   *  - by 3U to account for 40% table density at expansion (typical),
   *  - by 2U to account for worst-case due to doubling strategy for increasing bucket array size
   * requires size multiplied
   *  - by 8U to account for current AVL_LB allocator alignment requirements
   */
  pool = _kvstore->create_pool(pool_name(), ( many_count_target * 64U * 3U * 2U + 4 * single_value_size ) * 8U, 0, estimated_object_count);
  ASSERT_LT(0, int64_t(pool));
}

TEST_F(KVStore_test, BasicGet0)
{
  void * value = nullptr;
  size_t value_len = 0;

  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_NE(IKVStore::S_OK, r);
  if( r == IKVStore::S_OK )
  {
    ASSERT_EQ("Key already exists", "Did you forget to delete the pool before running the test?");
  }
  _kvstore->free_memory(value);
}

TEST_F(KVStore_test, BasicPut)
{
  single_value.resize(single_value_size);

  auto r = _kvstore->put(pool, single_key, single_value.data(), single_value.length());
  EXPECT_EQ(IKVStore::S_OK, r);
}

TEST_F(KVStore_test, BasicPutLocked)
{
  single_value.resize(single_value_size);
  void *value0 = nullptr;
  std::size_t value0_len = 0;
  auto *lk = _kvstore->lock(pool, single_key, IKVStore::STORE_LOCK_READ, value0, value0_len);
  EXPECT_NE(nullptr, lk);
  auto r = _kvstore->put(pool, single_key, single_value.data(), single_value.length());
  EXPECT_EQ(IKVStore::E_ALREADY_EXISTS, r);
  r = _kvstore->unlock(pool, lk);
  EXPECT_EQ(IKVStore::S_OK, r);
}

TEST_F(KVStore_test, BasicGet1)
{
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(IKVStore::S_OK, r);
  if ( IKVStore::S_OK == r )
  {
    PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
    EXPECT_EQ(0, memcmp(single_value.data(), value, single_value.size()));
    _kvstore->free_memory(value);
  }
}

/* hstore issue 41 specifies different implementations for same-size replace vs different-size replace. */
TEST_F(KVStore_test, BasicReplaceSameSize)
{
  {
    single_value_updated_same_size.resize(single_value_size);
    auto r = _kvstore->put(pool, single_key, single_value_updated_same_size.data(), single_value_updated_same_size.length());
    EXPECT_EQ(IKVStore::S_OK, r);
  }
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(IKVStore::S_OK, r);
  if ( IKVStore::S_OK == r )
  {
    PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
    EXPECT_EQ(0, memcmp(single_value_updated_same_size.data(), value, single_value_updated_same_size.size()));
    _kvstore->free_memory(value);
  }
}

TEST_F(KVStore_test, BasicReplaceDifferentSize)
{
  {
    auto r = _kvstore->put(pool, single_key, single_value_updated_different_size.data(), single_value_updated_different_size.length());
    EXPECT_EQ(IKVStore::S_OK, r);
  }
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(IKVStore::S_OK, r);
  if ( IKVStore::S_OK == r )
  {
    PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
    EXPECT_EQ(0, memcmp(single_value_updated_different_size.data(), value, single_value_updated_different_size.size()));
    _kvstore->free_memory(value);
  }
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
    if ( IKVStore::S_OK == _kvstore->get(pool, key, old_value, old_value_len) )
    {
      _kvstore->free_memory(old_value);
      ++extant_count;
    }
    else
    {
      auto r = _kvstore->put(pool, key, value.data(), value.length());
      EXPECT_EQ(IKVStore::S_OK, r);
      if ( r == IKVStore::S_OK )
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
  auto value_len_sum = 0;
  _kvstore->map(pool,[&value_len_sum](const std::string &key,
                        const void * value,
                        const size_t value_len) -> int
                {
					value_len_sum += value_len;
                    return 0;
                  });
  EXPECT_EQ(single_value_updated_different_size.length() + many_count_actual * many_value_length, value_len_sum);
}

TEST_F(KVStore_test, BasicMapKeys)
{
  auto key_len_sum = 0;
  _kvstore->map_keys(pool,[&key_len_sum](const std::string &key) -> int
                {
					key_len_sum += key.size();
                    return 0;
                  });
  EXPECT_EQ(single_key.size() + many_count_actual * many_key_length, key_len_sum);
}

TEST_F(KVStore_test, Count1)
{
  auto count = _kvstore->count(pool);
  /* count should reflect Put, PutMany */
  EXPECT_EQ(single_count + many_count_actual, count);
}

TEST_F(KVStore_test, CountByBucket)
{
  std::uint64_t count = 0;
  _kvstore->debug(pool, 2 /* COUNT_BY_BUCKET */, reinterpret_cast<std::uint64_t>(&count));
  /* should reflect Put, PutMany */
  EXPECT_EQ(single_count + many_count_actual, count);
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
    pool = _kvstore->open_pool(pool_name(), 0);
  }
  ASSERT_LT(0, int64_t(pool));
}

TEST_F(KVStore_test, Size2a)
{
  auto count = _kvstore->count(pool);
  /* count should reflect Put, PutMany */
  EXPECT_EQ(single_count + many_count_actual, count);
}

TEST_F(KVStore_test, BasicGet2)
{
  void * value = nullptr;
  size_t value_len = 0;
  auto r = _kvstore->get(pool, single_key, value, value_len);
  EXPECT_EQ(IKVStore::S_OK, r);
  if ( IKVStore::S_OK == r )
  {
    PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
    _kvstore->free_memory(value);
  }
}

TEST_F(KVStore_test, BasicGetAttribute)
{
  std::vector<uint64_t> attr;
  auto r = _kvstore->get_attribute(pool, IKVStore::VALUE_LEN, attr, &single_key);
  EXPECT_EQ(IKVStore::S_OK, r);
  if ( IKVStore::S_OK == r )
  {
    EXPECT_EQ(1, attr.size());
    if ( 1 == attr.size() )
    {
      EXPECT_EQ(attr[0], single_value_updated_different_size.length());
    }
  }
  r = _kvstore->get_attribute(pool, Component::IKVStore::Attribute(0), attr, &single_key);
  EXPECT_EQ(IKVStore::E_NOT_SUPPORTED, r);
  r = _kvstore->get_attribute(pool, IKVStore::VALUE_LEN, attr, nullptr);
  EXPECT_EQ(IKVStore::E_BAD_PARAM, r);
  r = _kvstore->get_attribute(pool, IKVStore::VALUE_LEN, attr, &missing_key);
  EXPECT_EQ(IKVStore::E_KEY_NOT_FOUND, r);
}

TEST_F(KVStore_test, ResizeAttribute)
{
  std::vector<uint64_t> attr;

  auto r = _kvstore->get_attribute(pool, IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr);
  EXPECT_EQ(IKVStore::S_OK, r);
  ASSERT_EQ(1, attr.size());
  EXPECT_EQ(1, attr[0]);

  attr[0] = false;
  r = _kvstore->set_attribute(pool, IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr);
  EXPECT_EQ(IKVStore::S_OK, r);
  EXPECT_EQ(1, attr.size());

  attr.clear();
  r = _kvstore->get_attribute(pool, IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr);
  EXPECT_EQ(IKVStore::S_OK, r);
  ASSERT_EQ(1, attr.size());
  EXPECT_EQ(0, attr[0]);

  attr[0] = 34;
  r = _kvstore->set_attribute(pool, IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr);
  EXPECT_EQ(IKVStore::S_OK, r);
  EXPECT_EQ(1, attr.size());

  attr.clear();
  r = _kvstore->get_attribute(pool, IKVStore::AUTO_HASHTABLE_EXPANSION, attr, nullptr);
  EXPECT_EQ(IKVStore::S_OK, r);
  ASSERT_EQ(1, attr.size());
  EXPECT_EQ(1, attr[0]);
}

TEST_F(KVStore_test, Size2b)
{
  auto count = _kvstore->count(pool);
  /* count should reflect PutMany */
  EXPECT_EQ(single_count + many_count_actual, count);
}

TEST_F(KVStore_test, GetMany)
{
  for ( auto i = 0; i != get_expand; ++i )
  {
    std::size_t mismatch_count = 0;
    for ( auto &kv : kvv )
    {
      const auto &key = std::get<0>(kv);
      const auto &ev = std::get<1>(kv);
      char value[many_value_length * 2];
      std::size_t value_len = many_value_length * 2;
      void *vp = value;
      auto r = _kvstore->get(pool, key, vp, value_len);
      EXPECT_EQ(IKVStore::S_OK, r);
      if ( IKVStore::S_OK == r )
      {
        EXPECT_EQ(vp, (void *)value);
        EXPECT_EQ(ev.size(), value_len);
        mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
      }
    }
    EXPECT_EQ(extant_count, mismatch_count);
  }
}

TEST_F(KVStore_test, GetManyAllocating)
{
  for ( auto i = 0; i != get_expand; ++i )
  {
    std::size_t mismatch_count = 0;
    for ( auto &kv : kvv )
    {
      const auto &key = std::get<0>(kv);
      const auto &ev = std::get<1>(kv);
      void * value = nullptr;
      std::size_t value_len = 0;
      auto r = _kvstore->get(pool, key, value, value_len);
      EXPECT_EQ(IKVStore::S_OK, r);
      if ( IKVStore::S_OK == r )
      {
        EXPECT_EQ(ev.size(), value_len);
        mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
        _kvstore->free_memory(value);
      }
    }
    EXPECT_EQ(extant_count, mismatch_count);
  }
}

TEST_F(KVStore_test, GetDirectMany)
{
  for ( auto i = 0; i != get_expand; ++i )
  {
    std::size_t mismatch_count = 0;
    for ( auto &kv : kvv )
    {
      const auto &key = std::get<0>(kv);
      const auto &ev = std::get<1>(kv);
      char value[many_value_length * 2];
      size_t value_len = many_value_length * 2;
      auto r = _kvstore->get_direct(pool, key, value, value_len);
      EXPECT_EQ(IKVStore::S_OK, r);
      if ( IKVStore::S_OK == r )
      {
        EXPECT_EQ(ev.size(), value_len);
        mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
      }
    }
    EXPECT_EQ(extant_count, mismatch_count);
  }
}

TEST_F(KVStore_test, GetRegions)
{
  std::vector<::iovec> v;
  auto r = _kvstore->get_pool_regions(pool, v);
  EXPECT_EQ(IKVStore::S_OK, r);
  if ( IKVStore::S_OK == r )
  {
    EXPECT_EQ(1, v.size());
    if ( 1 == v.size() )
    {
      std::cerr << "Pool region at " << v[0].iov_base << " len " << v[0].iov_len << "\n";
      auto iov_base = reinterpret_cast<std::uintptr_t>(v[0].iov_base);
      EXPECT_EQ(iov_base & 0xfff, 0);
      EXPECT_GT(v[0].iov_len, many_count_target * 64U * 3U * 2U);
      EXPECT_LT(v[0].iov_len, GB(512));
    }
  }
}

TEST_F(KVStore_test, LockMany)
{
  /* Lock for read (should succeed)
   * Lock again for read (should succeed).
   * Lock for write (should fail).
   * Lock a non-exisetent key for write (should succeed, creating the key).
   *
   * Undo the three successful locks.
   */
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
    EXPECT_NE(nullptr, r0);
    if ( nullptr != r0 )
    {
      EXPECT_EQ(many_value_length, value0_len);
      EXPECT_EQ(0, memcmp(ev.data(), value0, ev.size()));
    }
    void * value1 = nullptr;
    std::size_t value1_len = 0;
    auto r1 = _kvstore->lock(pool, key, IKVStore::STORE_LOCK_READ, value1, value1_len);
    EXPECT_NE(nullptr, r1);
    if ( nullptr != r1 )
    {
      EXPECT_EQ(many_value_length, value1_len);
      EXPECT_EQ(0, memcmp(ev.data(), value1, ev.size()));
    }
    /* Exclusive locking test. */

    void * value2 = nullptr;
    std::size_t value2_len = 0;
    auto r2 = _kvstore->lock(pool, key, IKVStore::STORE_LOCK_WRITE, value2, value2_len);
    EXPECT_EQ(nullptr, r2);

    void * value3 = nullptr;
    std::size_t value3_len = many_value_length;
    auto r3 = _kvstore->lock(pool, key_new, IKVStore::STORE_LOCK_WRITE, value3, value3_len);
    EXPECT_NE(nullptr, r3);
    if ( nullptr != r3 )
    {
      EXPECT_EQ(many_value_length, value3_len);
      EXPECT_NE(nullptr, value3);
    }

    if ( nullptr != r0 )
    {
      auto r0x = _kvstore->unlock(pool, r0);
      EXPECT_EQ(IKVStore::S_OK, r0x);
    }
    if ( nullptr != r1 )
    {
      auto r1x = _kvstore->unlock(pool, r1);
      EXPECT_EQ(IKVStore::S_OK, r1x);
    }
    if ( nullptr != r3 )
    {
      auto r3x = _kvstore->unlock(pool, r3);
      EXPECT_EQ(IKVStore::S_OK, r3x);
    }

    ++ct;
  }
}

TEST_F(KVStore_test, Size2c)
{
  auto count = _kvstore->count(pool);
  /* count should reflect Put, PutMany and LockMany */
  EXPECT_EQ(single_count + many_count_actual + lock_count, count);
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
    EXPECT_EQ(IKVStore::S_OK, r);
  }

  {
    void * value = nullptr;
    size_t value_len = 0;
    auto r = _kvstore->get(pool, single_key, value, value_len);
    EXPECT_EQ(IKVStore::S_OK, r);
    if ( IKVStore::S_OK == r )
    {
      PINF("Value=(%.*s) %zu", static_cast<int>(value_len), static_cast<char *>(value), value_len);
      EXPECT_EQ(single_value_updated_different_size.size(), value_len);
      EXPECT_EQ(0, memcmp(single_value_updated3.data(), value, single_value_updated3.size()));
      _kvstore->free_memory(value);
    }
  }

  auto count = _kvstore->count(pool);
  EXPECT_EQ(single_count + many_count_actual + lock_count, count);
}

TEST_F(KVStore_test, BasicErase)
{
  {
    auto r = _kvstore->erase(pool, single_key);
    EXPECT_EQ(IKVStore::S_OK, r);
  }

  auto count = _kvstore->count(pool);
  EXPECT_EQ(many_count_actual + lock_count, count);
}

TEST_F(KVStore_test, EraseMany)
{
  auto erase_count = 0;
  for ( auto &kv : kvv )
  {
    const auto &key = std::get<0>(kv);
    auto r = _kvstore->erase(pool, key);
    if ( r == IKVStore::S_OK )
    {
      ++erase_count;
    }
  }
  EXPECT_LE(many_count_actual, erase_count);
  auto count = _kvstore->count(pool);
  EXPECT_EQ(lock_count, count);
}

TEST_F(KVStore_test, DeletePool)
{
  _kvstore->close_pool(pool);
  _kvstore->delete_pool(pool_name());
}

} // namespace

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
