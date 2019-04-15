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
#include <common/utils.h>
#include <api/components.h>
/* note: we do not include component source, only the API definition */
#include <api/kvstore_itf.h>

#if defined HAS_PROFILER
#include <gperftools/profiler.h> /* Alas, no __has_include until C++17 */
#else
int ProfilerStart(const char *) {}
void ProfilerStop() {}
#endif

#include <chrono>
#include <cstdlib> /* getenv */
#include <functional> /* function */
#include <future>
#include <iostream>
#include <string>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

class timer
{
public:
	using clock_t = std::chrono::steady_clock;
	using duration_t = typename clock_t::duration;
private:
	std::function<void(duration_t) noexcept> _f;
	clock_t::time_point _start;
public:
	timer(std::function<void(duration_t) noexcept> f_)
		: _f(f_)
		, _start(clock_t::now())
	{}
	~timer()
	{
		_f(clock_t::now() - _start);
	}
};

/*
 * Performancte test:
 *
 * export PMEM_IS_PMEM_FORCE=1
 * LD_PRELOAD=/usr/lib/libprofiler.so CPUPROFILE=cpu.profile ./src/components/hstore/unit_test/hstore-test3
 *
 * memory requirement:
 *   memory (MB)  entries   key_sz  value_sz
 *    MB(128)       67484     8        16
 *    MB(512)      459112     8        16
 *    MB(2048UL)  2028888     8        16
 *    MB(2048UL)   918925    16        32
 *    MB(4096UL) >2000000    16        32
 *
 * (this table obsoleted, or at least made harder to generate, by use of multiple key and data sizes)
 */

using namespace Component;

namespace {

// The fixture for testing class Foo.
class KVStore_test
  : public ::testing::Test
{

  static const std::size_t many_count_target_large;
  static const std::size_t estimated_object_count_large;
  /* Shorter test: use when PMEM_IS_PMEM_FORCE=0 */
  static const std::size_t many_count_target_small;
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

  static constexpr std::size_t threads = 4;
  using poolv_t = std::array<Component::IKVStore::pool_t, threads>;

  // Objects declared here can be used by all tests in the test case

  /* persistent memory if enabled at all, is simulated and not real */
  static bool pmem_simulated;
  static Component::IKVStore * _kvstore;

  static constexpr unsigned many_key_length_short = 16;
  static constexpr unsigned many_key_length_long = 32;
  static constexpr unsigned many_value_length_short = 16;
  static constexpr unsigned many_value_length_long = 128;
  using kv_t = std::tuple<std::string, std::string>;
  using kvv_t = std::vector<std::tuple<std::string, std::string>>;
  /* data (puts in particular) are tested in 3 combinations:
   * short short (key and data both fit in the hash table space)
   * short long (key fits in hash table space, data needs a memory allocation)
   * long long (key and data both need memory allocations)
   *
   * The current limit for fit in hash table is 23 bytes, for both key and data.
   */
  static kvv_t kvv_short_short;
  static kvv_t kvv_short_long;
  static kvv_t kvv_long_long;

  static std::size_t multi_count_actual;
  static std::size_t estimated_object_count;
  static std::size_t many_count_target;
  static auto populate_many(char tag, std::size_t key_length, std::size_t value_length) -> kvv_t;
  static long unsigned put_many(Component::IKVStore::pool_t pool, const kvv_t &kvv, const std::string &descr);
  static long unsigned put_many_threaded(const kvv_t &kvv, const std::string &descr);
  static void get_many(Component::IKVStore::pool_t pool, const kvv_t &kvv, const std::string &descr);
  static void get_many_threaded(const kvv_t &kvv, const std::string &descr);

  std::string pool_name(int i) const
  {
    return "test-" + store_map::impl->name + store_map::numa_zone() + "-" + std::to_string(i) + ".pool";
  }
  static poolv_t pool;
};

constexpr std::size_t KVStore_test::estimated_object_count_small;
const std::size_t KVStore_test::many_count_target_large = std::getenv("COUNT_TARGET") ? std::stoul(std::getenv("COUNT_TARGET")) : 2000000;
const std::size_t KVStore_test::estimated_object_count_large = many_count_target_large;
const std::size_t KVStore_test::many_count_target_small = std::getenv("COUNT_TARGET") ? std::stoul(std::getenv("COUNT_TARGET")) : 400;

bool KVStore_test::pmem_simulated = getenv("PMEM_IS_PMEM_FORCE");
Component::IKVStore *KVStore_test::_kvstore;
KVStore_test::poolv_t KVStore_test::pool;

constexpr unsigned KVStore_test::many_key_length_short;
constexpr unsigned KVStore_test::many_key_length_long;
constexpr unsigned KVStore_test::many_value_length_short;
constexpr unsigned KVStore_test::many_value_length_long;
KVStore_test::kvv_t KVStore_test::kvv_short_short;
KVStore_test::kvv_t KVStore_test::kvv_short_long;
KVStore_test::kvv_t KVStore_test::kvv_long_long;

std::size_t KVStore_test::multi_count_actual = 0;
std::size_t KVStore_test::estimated_object_count = KVStore_test::pmem_simulated ? estimated_object_count_small : estimated_object_count_large;
std::size_t KVStore_test::many_count_target = KVStore_test::pmem_simulated ? many_count_target_small : many_count_target_large;

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
    for ( auto i = 0; i != pool.size(); ++i )
    {
      try
      {
        _kvstore->delete_pool(pool_name(i));
      }
      catch ( Exception & )
      {
      }
    }
  }
}

TEST_F(KVStore_test, CreatePools)
{
  /* time to create pools, in estimate objects / second */
  ASSERT_TRUE(_kvstore);
  timer t(
    [] (timer::duration_t d) {
      auto seconds = std::chrono::duration<double>(d).count();
      std::cerr << "create pool" << " " << estimated_object_count * pool.size() / seconds << " estimated objects per second\n";
    }
  );
  for ( auto i = 0; i != pool.size(); ++i )
  {
    pool[i] = _kvstore->create_pool(pool_name(i), GB(12UL), 0, estimated_object_count);
    ASSERT_LT(0, int64_t(pool[i]));
  }
}

auto KVStore_test::populate_many(const char tag, std::size_t key_length, std::size_t value_length) -> kvv_t
{
  std::mt19937_64 r0{};
  kvv_t kvv;
  for ( auto i = 0; i != many_count_target; ++i )
  {
    auto ukey = r0();
    std::ostringstream s;
    s << tag << std::hex << ukey;
    auto key = s.str();
    key.resize(key_length, '.');
    auto value = std::to_string(i);
    value.resize(value_length, '.');
    kvv.emplace_back(key, value);
  }
  return kvv;
}

TEST_F(KVStore_test, PopulateManySmallSmall)
{
  kvv_short_short = populate_many('A', many_key_length_short, many_value_length_short);
}

TEST_F(KVStore_test, PopulateManySmallLarge)
{
  kvv_short_long = populate_many('B', many_key_length_short, many_value_length_long);
}

TEST_F(KVStore_test, PopulateManyLargeLarge)
{
  kvv_long_long = populate_many('C', many_key_length_long, many_value_length_long);
}

long unsigned KVStore_test::put_many(Component::IKVStore::pool_t pool, const kvv_t &kvv, const std::string &descr)
{
  long unsigned count = 0;
  {
    timer t(
      [&count, &descr] (timer::duration_t d) {
        double seconds = std::chrono::duration_cast<std::chrono::microseconds>(d).count() / 1e6;
        std::cerr << descr << " " << count / seconds << " per second\n";
      }
    );
    for ( auto &kv : kvv )
    {
      const auto &key = std::get<0>(kv);
      const auto &value = std::get<1>(kv);
      auto r = _kvstore->put(pool, key, value.data(), value.length());
      if ( r == S_OK )
      {
          ++count;
      }
      else
      {
         std::cerr << __func__ << " FAIL " << key << "\n";
      }
    }
  }
  return count;
}

long unsigned KVStore_test::put_many_threaded(const kvv_t &kvv, const std::string &descr)
{

  std::vector<std::future<long unsigned>> v;
  ProfilerStart(("test4-put-" + descr + "-cpu-" + store_map::impl->name + ".profile").c_str());
  for ( auto p : pool )
  {
    v.emplace_back(std::async(std::launch::async, put_many, p, kvv, descr));
  }

  long unsigned count_actual = 0;
  for ( auto &e : v ) { count_actual += e.get(); }
  ProfilerStop();
  return count_actual;
}

TEST_F(KVStore_test, PutManyShortShort)
{
  ASSERT_NE(nullptr, _kvstore);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  auto count_actual = put_many_threaded(kvv_short_short, "short_short");

  EXPECT_GE(many_count_target*pool.size(), count_actual);
  EXPECT_LE(many_count_target*pool.size() * 0.99, double(count_actual));

  multi_count_actual += count_actual;
}

TEST_F(KVStore_test, PutManyShortLong)
{
  ASSERT_NE(nullptr, _kvstore);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  auto count_actual = put_many_threaded(kvv_short_long, "short_long");

  EXPECT_GE(many_count_target*pool.size(), count_actual);
  EXPECT_LE(many_count_target*pool.size() * 0.99, double(count_actual));

  multi_count_actual += count_actual;
}

TEST_F(KVStore_test, PutManyLongLong)
{
  ASSERT_NE(nullptr, _kvstore);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  auto count_actual = put_many_threaded(kvv_long_long, "long_long");

  EXPECT_GE(many_count_target*pool.size(), count_actual);
  EXPECT_LE(many_count_target*pool.size() * 0.99, double(count_actual));

  multi_count_actual += count_actual;
}

TEST_F(KVStore_test, Size1)
{
  ASSERT_NE(nullptr, _kvstore);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  unsigned long count = 0;
  for ( auto p : pool )
  {
    count += _kvstore->count(p);
  }

  /* count should reflect PutMany */
  EXPECT_EQ(multi_count_actual, count);
}

void KVStore_test::get_many(Component::IKVStore::pool_t pool, const kvv_t &kvv, const std::string &descr)
{
  /* get is quick; run 10 for better profiling */
  {
    unsigned amplification = 10;
	auto ct = amplification * kvv.size();
    timer t(
		[&descr,ct] (timer::duration_t d) {
			double seconds = std::chrono::duration_cast<std::chrono::microseconds>(d).count() / 1e6;
			std::cerr << descr << " " << ct << " in " << seconds << " => " << ct / seconds << " per second\n";
		}
	);
    for ( auto i = 0; i != amplification; ++i )
    {
      for ( auto &kv : kvv )
      {
        const auto &key = std::get<0>(kv);
        void * value = nullptr;
        size_t value_len = 0;
        auto r = _kvstore->get(pool, key, value, value_len);
        EXPECT_EQ(S_OK, r);
        EXPECT_EQ(std::get<1>(kv).size(), value_len);
        _kvstore->free_memory(value);
      }
    }
  }
}

void KVStore_test::get_many_threaded(const kvv_t &kvv, const std::string &descr)
{
  std::vector<std::future<void>> v;
  ProfilerStart(("test4-get-" + descr + "-cpu-" + store_map::impl->name + ".profile").c_str());
  for ( auto p : pool )
  {
    v.emplace_back(std::async(std::launch::async, get_many, p, kvv, descr));
  }
  for ( auto &e : v ) { e.get(); }
  ProfilerStop();
}

TEST_F(KVStore_test, GetManyShortShort)
{
  ASSERT_NE(nullptr, _kvstore);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  get_many_threaded(kvv_short_short, "short_short");
}

TEST_F(KVStore_test, GetManyShortLong)
{
  ASSERT_NE(nullptr, _kvstore);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  get_many_threaded(kvv_short_long, "short_long");
}

TEST_F(KVStore_test, GetManyLongLong)
{
  ASSERT_NE(nullptr, _kvstore);
  for ( auto p : pool ) { ASSERT_LT(0, int64_t(p)); }

  get_many_threaded(kvv_long_long, "long_long");
}

TEST_F(KVStore_test, ClosePool)
{
  timer t(
    [] (timer::duration_t d) {
      auto seconds = std::chrono::duration<double>(d).count();
      std::cerr << "close pool" << " " << multi_count_actual / seconds << " objects per second\n";
    }
  );
  for ( auto p : pool )
  {
    if ( _kvstore && 0 < int64_t(p) )
    {
      _kvstore->close_pool(p);
    }
  }
}

TEST_F(KVStore_test, OpenPool2)
{
  timer t(
    [] (timer::duration_t d) {
      auto seconds = std::chrono::duration<double>(d).count();
      std::cerr << "open pool" << " " << multi_count_actual << "/" << seconds << " = " << multi_count_actual / seconds << " objects per second\n";
    }
  );
  ASSERT_TRUE(_kvstore);
  for ( auto i = 0; i != pool.size(); ++i )
  {
    pool[i] = _kvstore->open_pool(pool_name(i));
    ASSERT_LT(0, int64_t(pool[i]));
  }
}

TEST_F(KVStore_test, ClosePool2)
{
  for ( auto p : pool )
  {
    if ( _kvstore && 0 < int64_t(p) )
    {
      _kvstore->close_pool(p);
    }
  }
}

TEST_F(KVStore_test, DeletePool)
{
  for ( auto i = 0; i != pool.size(); ++i )
  {
    if ( _kvstore )
    {
      try
      {
        _kvstore->delete_pool(pool_name(i));
      }
      catch ( ... )
      {
      }
    }
  }
}

} // namespace

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
