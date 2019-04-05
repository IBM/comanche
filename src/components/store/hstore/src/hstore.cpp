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


#include "hstore.h"

#define USE_PMEM 0

/*
 * USE_PMEM 1
 *   USE_CC_HEAP 0: allocation from pmemobj pool
 *   USE_CC_HEAP 1: simple allocation using actual addresses from a large region obtained from pmemobj 
 *   USE_CC_HEAP 2: simple allocation using offsets from a large region obtained from pmemobj 
 *   USE_CC_HEAP 3: AVL-based allocation using actual addresses from a large region obtained from pmemobj 
 * USE_PMEM 0
 *   USE_CC_HEAP 1: simple allocation using actual addresses from a large region obtained from dax_map 
 *   USE_CC_HEAP 2: simple allocation using offsets from a large region obtained from dax_map (NOT TESTED)
 *   USE_CC_HEAP 3: AVL-based allocation using actual addresses from a large region obtained from dax_map 
 *
 */
#if USE_PMEM
/* with PMEM, choose the CC_HEAP version: 0, 1, 2, 3 */
#define USE_CC_HEAP 0
#else
/* without PMEM, only heap version 1 or 3 works */
#define USE_CC_HEAP 3
#endif

#if USE_CC_HEAP == 1
#include "allocator_cc.h"
#elif USE_CC_HEAP == 2
#include "allocator_co.h"
#elif USE_CC_HEAP == 3
#include "allocator_rc.h"
#endif
#include "atomic_controller.h"
#include "hop_hash.h"
#include "perishable.h"
#include "persist_fixed_string.h"

#include <stdexcept>
#include <set>

#include <city.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/utils.h>

#if USE_PMEM
#include "hstore_pmem_types.h"
#include "persister_pmem.h"
#else
#include "hstore_nupm_types.h"
#include "persister_nupm.h"
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <tbb/scalable_allocator.h> /* scalable_free */
#pragma GCC diagnostic pop

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring> /* strerror, memcpy */
#include <memory> /* unique_ptr */
#include <new>
#include <map> /* session set */
#include <mutex> /* thread safe use of libpmempool/obj */
#include <stdexcept> /* domain_error */

#define PREFIX "HSTORE : %s: "

#if 0
/* thread-safe hash */
#include <mutex>
using hstore_shared_mutex = std::shared_timed_mutex;
static constexpr auto thread_model = Component::IKVStore::THREAD_MODEL_MULTI_PER_POOL;
static constexpr auto is_thread_safe = true;
#else
/* not a thread-safe hash */
#include "dummy_shared_mutex.h"
using hstore_shared_mutex = dummy::shared_mutex;
static constexpr auto thread_model = Component::IKVStore::THREAD_MODEL_SINGLE_PER_POOL;
static constexpr auto is_thread_safe = false;
#endif

template<typename T>
  struct type_number;

template<> struct type_number<char> { static constexpr uint64_t value = 2; };

namespace
{
  constexpr bool option_DEBUG = false;
  namespace type_num
  {
    constexpr uint64_t persist = 1U;
    constexpr uint64_t heap = 2U;
  }
}

#if USE_CC_HEAP == 1
using ALLOC_T = allocator_cc<char, Persister>;
#elif USE_CC_HEAP == 2
using ALLOC_T = allocator_co<char, Persister>;
#elif USE_CC_HEAP == 3
using ALLOC_T = allocator_rc<char, Persister>;
#else /* USE_CC_HEAP */
using ALLOC_T = allocator_pobj_cache_aligned<char>;
#endif /* USE_CC_HEAP */

using DEALLOC_T = typename ALLOC_T::deallocator_type;
using KEY_T = persist_fixed_string<char, DEALLOC_T>;
using MAPPED_T = persist_fixed_string<char, DEALLOC_T>;

struct pstr_hash
{
  using argument_type = KEY_T;
  using result_type = std::uint64_t;
  static result_type hf(const argument_type &s)
  {
    return CityHash64(s.data(), s.size());
  }
};

using HASHER_T = pstr_hash;

using allocator_segment_t = ALLOC_T::rebind<std::pair<const KEY_T, MAPPED_T>>::other;
using allocator_atomic_t = ALLOC_T::rebind<impl::mod_control>::other;

#if USE_CC_HEAP == 1
#elif USE_CC_HEAP == 2
#else
template<> struct type_number<impl::mod_control> { static constexpr std::uint64_t value = 4; };
#endif /* USE_CC_HEAP */

using table_t =
  table<
  KEY_T
  , MAPPED_T
  , HASHER_T
  , std::equal_to<KEY_T>
  , allocator_segment_t
  , hstore_shared_mutex
  >;

template<> struct type_number<table_t::value_type> { static constexpr std::uint64_t value = 5; };
template<> struct type_number<table_t::base::persist_data_t::bucket_aligned_t> { static constexpr uint64_t value = 6; };

using persist_data_t = typename impl::persist_data<allocator_segment_t, table_t::value_type>;

template <typename Handle, typename Allocator, typename Table>
  class session;
#if USE_PMEM
#include "hstore_pmem.h"
using session_t = session<hstore_pmem::open_pool_handle, ALLOC_T, table_t>;
#else
#include "hstore_nupm.h"
using session_t = session<hstore_nupm::open_pool_handle, ALLOC_T, table_t>;
#endif

/* globals */

thread_local std::set<tracked_pool *> tls_cache = {};

auto hstore::locate_session(const Component::IKVStore::pool_t pid) -> tracked_pool *
{
  return dynamic_cast<tracked_pool *>(this->locate_open_pool(pid));
}

auto hstore::locate_open_pool(const Component::IKVStore::pool_t pid) -> tracked_pool *
{
  auto *const s = reinterpret_cast<tracked_pool *>(pid);
  auto it = tls_cache.find(s);
  if ( it == tls_cache.end() )
  {
    std::unique_lock<std::mutex> sessions_lk(_pools_mutex);
    auto ps = _pools.find(s);
    if ( ps == _pools.end() )
    {
      return nullptr;
    }
    it = tls_cache.insert(ps->second.get()).first;
  }
  return *it;
}

auto hstore::move_pool(const Component::IKVStore::pool_t pid) -> std::unique_ptr<tracked_pool>
{
  auto *const s = reinterpret_cast<tracked_pool *>(pid);

  std::unique_lock<std::mutex> sessions_lk(_pools_mutex);
  auto ps = _pools.find(s);
  if ( ps == _pools.end() )
    {
      throw API_exception(PREFIX "invalid pool identifier %p", __func__, s);
    }

  tls_cache.erase(s);
  auto s2 = std::move(ps->second);
  _pools.erase(ps);
  return s2;
}

hstore::hstore(const std::string &owner, const std::string &name, std::unique_ptr<Devdax_manager> mgr_)
#if USE_PMEM
  : _pool_manager(std::make_shared<hstore_pmem>(owner, name, option_DEBUG))
#else
  : _pool_manager(std::make_shared<hstore_nupm>(owner, name, std::move(mgr_), option_DEBUG))
#endif
  , _pools_mutex{}
  , _pools{}
{
}

hstore::~hstore()
{
}

auto hstore::thread_safety() const -> int
{
  return thread_model;
}

int hstore::get_capability(const Capability cap) const
{
  switch (cap)
  {
  case Capability::POOL_DELETE_CHECK: /*< checks if pool is open before allowing delete */
    return false;
  case Capability::RWLOCK_PER_POOL:   /*< pools are locked with RW-lock */
    return false;
  case Capability::POOL_THREAD_SAFE:  /*< pools can be shared across multiple client threads */
    return is_thread_safe;
  default:
    return -1;
  }
}

auto hstore::create_pool(const std::string & name_,
                         const std::size_t size_,
                         std::uint32_t flags_,
                         const uint64_t  expected_obj_count_) -> pool_t
try
{
  std::cerr << "create_pool " << name_ << " size " << size_ << "\n";
  if ( option_DEBUG )
  {
    PLOG(PREFIX "pool_name=%s", __func__, name_.c_str());
  }
  {
    auto c = _pool_manager->pool_create_check(size_);
    if ( c != S_OK )  { return c; }
  }

  auto path = pool_path(name_);

  auto s =
    std::unique_ptr<session_t>(
      static_cast<session_t *>(
        _pool_manager->pool_create(path, size_, flags_ & ~(FLAGS_CREATE_ONLY|FLAGS_SET_SIZE), expected_obj_count_).release()
      )
    );

  auto p = s.get();
  std::unique_lock<std::mutex> sessions_lk(_pools_mutex);
  _pools.emplace(p, std::move(s));

  return reinterpret_cast<IKVStore::pool_t>(p);
}
catch ( const pool_error & )
{
  return flags_ & FLAGS_CREATE_ONLY
    ? static_cast<IKVStore::pool_t>(POOL_ERROR)
    : open_pool(name_, flags_ & ~FLAGS_SET_SIZE)
    ;
}

auto hstore::open_pool(const std::string &name_,
                       std::uint32_t flags) -> pool_t
{
  auto path = pool_path(name_);
  try {
    auto s = _pool_manager->pool_open(path, flags);
    auto p = static_cast<tracked_pool *>(s.get());
    std::unique_lock<std::mutex> sessions_lk(_pools_mutex);
    _pools.emplace(p, std::move(s));
    return reinterpret_cast<IKVStore::pool_t>(p);
  }
  catch( const pool_error & ) {
    return Component::IKVStore::POOL_ERROR;
  }
  catch( const std::invalid_argument & ) {
    return Component::IKVStore::POOL_ERROR;
  }
}

status_t hstore::close_pool(const pool_t pid)
{
  std::string path;
  try
  {
    auto pool = move_pool(pid);
    if ( option_DEBUG )
    {
      PLOG(PREFIX "closed pool (%" PRIxIKVSTORE_POOL_T ")", __func__, pid);
    }
  }
  catch ( const API_exception &e )
  {
    return E_INVAL;
  }
  _pool_manager->pool_close_check(path);
  return S_OK;
}

status_t hstore::delete_pool(const std::string &name_)
{
  auto path = pool_path(name_);

  _pool_manager->pool_delete(path);
  if ( option_DEBUG )
  {
    PLOG("pool deleted: %s", name_.c_str());
  }
  return S_OK;
}

auto hstore::put(const pool_t pool,
                 const std::string &key,
                 const void * value,
                 const std::size_t value_len,
                 std::uint32_t flags) -> status_t
{
  if ( option_DEBUG ) {
    PLOG(
         PREFIX "(key=%s) (value=%.*s)"
         , __func__
         , key.c_str()
         , int(value_len)
         , static_cast<const char*>(value)
         );
    assert(0 < value_len);
  }

  if ( (flags & ~FLAGS_DONT_STOMP) != 0 )
  {
    return E_BAD_PARAM;
  }
  if ( value == nullptr )
  {
    return E_BAD_PARAM;
  }

  const auto session = dynamic_cast<session_t *>(locate_session(pool));

  if ( session )
  {
    try
    {
      auto i = session->insert(key, value, value_len);

      return
        i.second                   ? S_OK
        : flags & FLAGS_DONT_STOMP ? E_KEY_EXISTS 
        : session->update_by_issue_41(key, value, value_len, i.first->second.data(), i.first->second.size())
        ;
    }
    catch ( std::bad_alloc & )
    {
      return E_FAIL;
    }
  }
  else
  {
    return E_POOL_NOT_FOUND;
  }
}

auto hstore::get_pool_regions(const pool_t pool, std::vector<::iovec>& out_regions) -> status_t
{
  const auto session = dynamic_cast<session_t *>(locate_session(pool));
  return session
    ? _pool_manager->pool_get_regions(session->pool(), out_regions)
    : E_POOL_NOT_FOUND
    ;
}

auto hstore::put_direct(const pool_t pool,
                        const std::string& key,
                        const void * value,
                        const std::size_t value_len,
                        memory_handle_t,
                        std::uint32_t flags) -> status_t
{
  return put(pool, key, value, value_len, flags);
}

auto hstore::get(const pool_t pool,
                 const std::string &key,
                 void*& out_value,
                 std::size_t& out_value_len) -> status_t
{
#if 0
  PLOG(PREFIX " get(%s)", __func__, key.c_str());
#endif
  const auto session = dynamic_cast<const session_t *>(locate_session(pool));
  return
    session
    ? session->get(key, out_value, out_value_len)
    : E_POOL_NOT_FOUND
    ;
}

auto hstore::get_direct(const pool_t pool,
                        const std::string & key,
                        void* out_value,
                        std::size_t& out_value_len,
                        Component::IKVStore::memory_handle_t) -> status_t
{
  const auto session = dynamic_cast<const session_t *>(locate_session(pool));
  return
    session
    ? session->get_direct(key, out_value, out_value_len)
    : E_POOL_NOT_FOUND
    ;
}

auto hstore::lock(const pool_t pool,
                  const std::string &key,
                  lock_type_t type,
                  void *& out_value,
                  std::size_t & out_value_len) -> key_t
{
  const auto session = dynamic_cast<session_t *>(locate_session(pool));
  return
    session
    ? session->lock(key, type, out_value, out_value_len)
    : KEY_NONE
    ;
}


auto hstore::unlock(const pool_t pool,
                    key_t key_) -> status_t
{
  const auto session = dynamic_cast<session_t *>(locate_session(pool));
  return
    session
    ? session->unlock(key_)
    : E_POOL_NOT_FOUND
    ;
}

auto hstore::apply(
                   const pool_t pool,
                   const std::string &key,
                   std::function<void(void*,std::size_t)> functor,
                   std::size_t object_size,
                   bool take_lock
                   ) -> status_t
{
  const auto apply_method = take_lock ? &session_t::lock_and_apply : &session_t::apply;
  const auto session = dynamic_cast<session_t *>(locate_session(pool));
  return
    session
    ? (session->*apply_method)(key, functor, object_size)
    : E_POOL_NOT_FOUND
    ;
}

auto hstore::erase(const pool_t pool,
                   const std::string &key
                   ) -> status_t
{
  const auto session = dynamic_cast<session_t *>(locate_session(pool));
  return session
    ? session->erase(key)
    : E_POOL_NOT_FOUND
    ;
}

std::size_t hstore::count(const pool_t pool)
{
  const auto session = dynamic_cast<session_t *>(locate_session(pool));
  return
    session
    ? session->count()
    : 0
    ;
}

void hstore::debug(const pool_t pool, const unsigned cmd, const uint64_t arg)
{
  switch ( cmd )
    {
    case 0:
      perishable::enable(bool(arg));
      break;
    case 1:
      perishable::reset(arg);
      break;
    case 2:
      {
        const auto session = dynamic_cast<const session_t *>(locate_session(pool));
        *reinterpret_cast<table_t::size_type *>(arg) = session ? session->bucket_count() : 0;
      }
      break;
    default:
      break;
    };
}

auto hstore::map(
                 pool_t pool,
                 std::function
                 <
                 int(const std::string &key, const void *val, std::size_t val_len)
                 > function
                 ) -> status_t
{
  const auto session = dynamic_cast<session_t *>(locate_session(pool));

  return session
    ? ( session->map(function), S_OK )
    : E_POOL_NOT_FOUND
    ;
}

auto hstore::free_memory(void * p) -> status_t
{
  scalable_free(p);
  return S_OK;
}

auto hstore::atomic_update(
                           const pool_t pool
                           , const std::string& key
                           , const std::vector<IKVStore::Operation *> &op_vector
                           , const bool take_lock) -> status_t
try
{
  const auto update_method = take_lock ? &session_t::lock_and_atomic_update : &session_t::atomic_update;
  const auto session = dynamic_cast<session_t *>(locate_session(pool));
    return
      session
      ? (session->*update_method)(key, op_vector)
      : E_POOL_NOT_FOUND
      ;
}
catch ( std::bad_alloc & )
{
  return E_FAIL;
}
catch ( std::system_error & )
{
  return E_FAIL;
}

