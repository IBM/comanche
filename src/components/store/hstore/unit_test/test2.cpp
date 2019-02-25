#include "store_map.h"

#include <gtest/gtest.h>
#include <common/utils.h>
#include <api/components.h>
/* note: we do not include component source, only the API definition */
#include <api/kvstore_itf.h>

#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace Component;

namespace {

// The fixture for testing class Foo.
class KVStore_test : public ::testing::Test {

  static constexpr std::size_t estimated_object_count_large = 64000000;
  /* More testing of table splits, at a performance cost */
  static constexpr std::size_t estimated_object_count_small = 1;

  static constexpr std::size_t many_count_target_large = 2000000;
  /* Shorter test: use when PMEM_IS_PMEM_FORCE=0 */
  static constexpr std::size_t many_count_target_small = 400;
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

  static const std::size_t estimated_object_count;

  static constexpr unsigned many_key_length = 8;
  static constexpr unsigned many_value_length = 16;
  using kv_t = std::tuple<std::string, std::string>;
  static std::vector<kv_t> kvv;
  static const std::size_t many_count_target;
  static std::size_t many_count_actual;

  std::string pool_dir() const
  {
    return "/mnt/pmem0/pool/0/";
  }

  std::string pool_name() const
  {
    return "test-" + store_map::impl->name + store_map::numa_zone() + ".pool";
  }
};

constexpr std::size_t KVStore_test::estimated_object_count_small;
constexpr std::size_t KVStore_test::estimated_object_count_large;
constexpr std::size_t KVStore_test::many_count_target_small;
constexpr std::size_t KVStore_test::many_count_target_large;

bool KVStore_test::pmem_simulated = getenv("PMEM_IS_PMEM_FORCE");
bool KVStore_test::pmem_effective = ! getenv("PMEM_IS_PMEM_FORCE") || getenv("PMEM_IS_PMEM_FORCE") == std::string("0");
Component::IKVStore * KVStore_test::_kvstore;

const std::size_t KVStore_test::estimated_object_count = pmem_simulated ? estimated_object_count_small : estimated_object_count_large;

constexpr unsigned KVStore_test::many_key_length;
constexpr unsigned KVStore_test::many_value_length;

const std::size_t KVStore_test::many_count_target = pmem_simulated ? many_count_target_small : many_count_target_large;
std::size_t KVStore_test::many_count_actual;
std::vector<KVStore_test::kv_t> KVStore_test::kvv;

TEST_F(KVStore_test, Instantiate)
{
  std::cerr
    << "PMEM " << (pmem_simulated ? "simulated" : "not simluated")
    << ", " << (pmem_effective ? "effective" : "not effective")
    << "\n";
  /* create object instance through factory */
  auto link_library = "libcomanche-" + store_map::impl->name + ".so";
  Component::IBase * comp = Component::load_component(link_library,
                                                      store_map::impl->factory_id);

  ASSERT_TRUE(comp);
  auto fact = static_cast<IKVStore_factory *>(comp->query_interface(IKVStore_factory::iid()));

  _kvstore = fact->create("owner", "name", store_map::location);

  fact->release_ref();
}

class pool_open
{
  Component::IKVStore *_kvstore;
  Component::IKVStore::pool_t _pool;
public:
  explicit pool_open(
    Component::IKVStore *kvstore_
    , const std::string& path_
    , const std::string& name_
    , unsigned int flags = 0
  )
    : _kvstore(kvstore_)
    , _pool(_kvstore->open_pool(path_, name_, flags))
  {
    if ( int64_t(_pool) < 0 )
    {
      throw std::runtime_error("Failed to open pool code " + std::to_string(-_pool));
    }
  }

  explicit pool_open(
    Component::IKVStore *kvstore_
    , const std::string& path_
    , const std::string& name_
    , const size_t size
    , unsigned int flags = 0
    , uint64_t expected_obj_count = 0
  )
    : _kvstore(kvstore_)
    , _pool(_kvstore->create_pool(path_, name_, size, flags, expected_obj_count))
  {}

  ~pool_open()
  {
    _kvstore->close_pool(_pool);
  }

  Component::IKVStore::pool_t pool() const noexcept { return _pool; }
};

TEST_F(KVStore_test, RemoveOldPool)
{
  if ( _kvstore )
  {
    try
    {
      _kvstore->delete_pool(pool_dir(), pool_name());
    }
    catch ( Exception & )
    {
    }
  }
}

TEST_F(KVStore_test, CreatePool)
{
  ASSERT_TRUE(_kvstore);
  pool_open p(_kvstore, pool_dir(), pool_name(), GB(15UL), 0, estimated_object_count);
  ASSERT_LT(0, int64_t(p.pool()));
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
  /* We will try the inserts many times, as the perishable timer will abort all but the last attempt */
  bool finished = false;

  _kvstore->debug(0, 0 /* enable */, 0);
  /*
   * We would like to generate "crashes" with some reasonable frequency,
   * but not at every store. (Every store would be too slow, at least
   * when using mmap to simulate persistent store). We use a Fibonacci
   * series to produce crashes at decreasingly frequent intervals.
   *
   * p0 and p1 produce a Fibonacci series in perishable_count
   */
  unsigned p0 = 0;
  unsigned p1 = 1;
  for (
    unsigned perishable_count = p0 + p1
    ; ! finished
    ; perishable_count = p0 + p1, p0 = p1, p1 = perishable_count
    )
  {
    unsigned extant_count = 0;
    unsigned fail_count = 0;
    unsigned succeed_count = 0;
    _kvstore->debug(0, 1 /* reset */, perishable_count);
    _kvstore->debug(0, 0 /* enable */, true);
    try
    {
      pool_open p(_kvstore, pool_dir(), pool_name());

      std::mt19937_64 r0{};

      for ( auto &kv : kvv )
      {
        const auto &key = std::get<0>(kv);
        const auto &value = std::get<1>(kv);
        void * old_value = nullptr;
        size_t old_value_len = 0;
        if ( S_OK == _kvstore->get(p.pool(), key, old_value, old_value_len) )
        {
          _kvstore->free_memory(old_value);
          ++extant_count;
        }
        else
        {
          auto r = _kvstore->put(p.pool(), key, value.c_str(), value.length());
          EXPECT_EQ(S_OK, r);
          if ( r == S_OK )
          {
            ++succeed_count;
          }
          else
          {
            ++fail_count;
          }
	}
      }
      EXPECT_EQ(many_count_target, extant_count + succeed_count + fail_count);
      many_count_actual = extant_count + succeed_count;
      finished = true;
      /* Done with forcing crashes */
      _kvstore->debug(0, 0 /* enable */, false);
      std::cerr << __func__ << " Final put pass " << perishable_count << " exists " << extant_count << " inserts " << succeed_count << " total " << many_count_actual << "\n";
    }
    catch ( const std::runtime_error &e )
    {
      if ( e.what() != std::string("perishable timer expired") ) { throw; }
      std::cerr << __func__ << " Perishable pass " << perishable_count << " exists " << extant_count << " inserts " << succeed_count << " total " << many_count_actual << "\n";
    }
  }
}

TEST_F(KVStore_test, GetMany)
{
  ASSERT_TRUE(_kvstore);
  if ( pmem_effective )
  {
    pool_open p(_kvstore, pool_dir(), pool_name());
    ASSERT_LT(0, int64_t(p.pool()));
    auto count = _kvstore->count(p.pool());
    {
      /* count should be close to PutMany many_count_actual; duplicate keys are the difference */
      EXPECT_LE(many_count_actual * 0.99, double(count));
    }
    {
      std::size_t mismatch_count = 0;
      for ( auto &kv : kvv )
      {
        const auto &key = std::get<0>(kv);
        const auto &ev = std::get<1>(kv);
        void * value = nullptr;
        size_t value_len = 0;
        auto r = _kvstore->get(p.pool(), key, value, value_len);
        EXPECT_EQ(S_OK, r);
        EXPECT_EQ(ev.size(), value_len);
        mismatch_count += ( ev.size() != value_len || 0 != memcmp(ev.data(), value, ev.size()) );
        _kvstore->free_memory(value);
      }
      /* We do not know exactly now many mismatches (caused by duplicates) to expcect,
       * because "extant_count" counts both extant items due to duplicate keys in the
       * population arrays and extant items due to restarts.
       * But it should be a small fraction of the total number of keys
       */
      EXPECT_GT(many_count_target * 0.01, double(mismatch_count));
    }

    {
      auto erase_count = 0;
      for ( auto &kv : kvv )
      {
        const auto &key = std::get<0>(kv);
        auto r = _kvstore->erase(p.pool(), key);
        if ( r == S_OK )
        {
          ++erase_count;
        }
      }
      EXPECT_EQ(count, erase_count);
      auto count = _kvstore->count(p.pool());
      EXPECT_EQ(0U, count);
    }
  }
}

TEST_F(KVStore_test, DeletePool)
{
  if ( pmem_effective )
  {
    _kvstore->delete_pool(pool_dir(), pool_name());
  }
}

} // namespace

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
