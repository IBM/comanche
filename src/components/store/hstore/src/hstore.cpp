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

#include "atomic_controller.h"
#include "hop_hash.h"
#include "perishable.h"
#include "persist_fixed_string.h"
#include "pool_path.h"

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
#include <cstring> /* strerror, memcmp, memcpy */
#include <memory> /* unique_ptr */
#include <new>
#include <map> /* session set */
#include <mutex> /* thread safe use of libpmempool/obj */
#include <stdexcept> /* domain_error */

#define PREFIX "HSTORE : %s: "

/*
 * To run hstore without PM, use variables USE_DRAM and NO_CLFLUSHOPT:
 *   USE_DRAM=24 NO_CLFLUSHOPT=1 DAX_RESET=1 ./dist/bin/kvstore-perf --test put --component hstore --path pools --pool_name foo --device_name /tmp/ --elements 1000000 --size 5000000000 --devices 0.0
 */

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
#elif USE_CC_HEAP == 2
#else
template<> struct type_number<impl::mod_control> { static constexpr std::uint64_t value = 4; };
#endif /* USE_CC_HEAP */

#if 0
template<> struct type_number<hstore::table_t::value_type> { static constexpr std::uint64_t value = 5; };
template<> struct type_number<hstore::table_t::base::persist_data_t::bucket_aligned_t> { static constexpr uint64_t value = 6; };
#endif

/* globals */

thread_local std::set<hstore::open_pool_t *> tls_cache = {};

auto hstore::locate_session(const Component::IKVStore::pool_t pid) -> open_pool_t *
{
  auto *const s = reinterpret_cast<open_pool_t *>(pid);
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

auto hstore::move_pool(const Component::IKVStore::pool_t pid) -> std::unique_ptr<open_pool_t>
{
  auto *const s = reinterpret_cast<open_pool_t *>(pid);

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

hstore::hstore(const std::string &owner, const std::string &name, std::unique_ptr<Devdax_manager> &&mgr_)
#if USE_PMEM
  : _pool_manager(std::make_shared<pm>(owner, name, option_DEBUG))
#else
  : _pool_manager(std::make_shared<pm>(owner, name, std::move(mgr_), option_DEBUG))
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

#include "hstore_session.h"

auto hstore::create_pool(const std::string & name_,
                         const std::size_t size_,
                         std::uint32_t flags_,
                         const uint64_t expected_obj_count_) -> pool_t
try
{
  if ( option_DEBUG )
  {
    PLOG(PREFIX "pool_name=%s size %zu", __func__, name_.c_str(), size_);
  }
  try
  {
    _pool_manager->pool_create_check(size_);
  }
  catch ( const std::exception & )
  {
    return Component::IKVStore::E_FAIL;
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
    auto p = static_cast<session_t *>(s.get());
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
    _pool_manager->pool_close_check(path);
  }
  catch ( const API_exception &e )
  {
    return Component::IKVStore::E_FAIL;
  }
  return Component::IKVStore::S_OK;
}

status_t hstore::delete_pool(const std::string &name_)
{
  auto path = pool_path(name_);

  _pool_manager->pool_delete(path);
  if ( option_DEBUG )
  {
    PLOG("pool deleted: %s", name_.c_str());
  }
  return Component::IKVStore::S_OK;
}

auto hstore::grow_pool(
  const pool_t pool
  , const std::size_t increment_size
  , std::size_t & reconfigured_size ) -> status_t
{
  const auto session = static_cast<session_t *>(locate_session(pool));
  if ( ! session )
  {
    return Component::IKVStore::E_POOL_NOT_FOUND;
  }
  try
  {
    reconfigured_size = session->pool_grow(_pool_manager->devdax_manager(), increment_size);
  }
  catch ( const std::bad_alloc & )
  {
    return Component::IKVStore::E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
  }
  return Component::IKVStore::S_OK;
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
    return Component::IKVStore::E_BAD_PARAM;
  }
  if ( value == nullptr )
  {
    return Component::IKVStore::E_BAD_PARAM;
  }

  const auto session = static_cast<session_t *>(locate_session(pool));

  if ( session )
  {
    try
    {
      auto i = session->insert(key, value, value_len);

      return
        i.second                   ? Component::IKVStore::S_OK
        : flags & FLAGS_DONT_STOMP ? Component::IKVStore::E_KEY_EXISTS
        : ( session->update_by_issue_41(key, value, value_len, i.first->second.data(), i.first->second.size()), Component::IKVStore::S_OK )
        ;
    }
    catch ( const std::bad_alloc & )
    {
      return Component::IKVStore::E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
    }
    catch ( const std::invalid_argument & )
    {
      return Component::IKVStore::E_NOT_SUPPORTED;
    }
  }
  else
  {
    return Component::IKVStore::E_POOL_NOT_FOUND;
  }
}

auto hstore::get_pool_regions(const pool_t pool, std::vector<::iovec>& out_regions) -> status_t
{
  const auto session = static_cast<session_t *>(locate_session(pool));
  if ( ! session )
  {
    return Component::IKVStore::E_POOL_NOT_FOUND;
  }
  out_regions = _pool_manager->pool_get_regions(session->handle());
  return Component::IKVStore::S_OK;
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
  const auto session = static_cast<const session_t *>(locate_session(pool));
  if ( ! session )
  {
    return Component::IKVStore::E_POOL_NOT_FOUND;
  }

  try
  {
    /* Although not documented, assume that non-zero
     * out_value implies that out_value_len holds
     * the buffer's size.
     */
    if ( out_value )
    {
      auto buffer_size = out_value_len;
      out_value_len = session->get(key, out_value, buffer_size);
      /*
       * It might be reasonable to
       *  a) fill the buffer and/or
       *  b) return the necessary size in out_value_len,
       * but neither action is documented, so we do not.
       */
      if ( buffer_size < out_value_len )
      {
        return Component::IKVStore::E_INSUFFICIENT_BUFFER;
      }
    }
    else
    {
      try
      {
        auto r = session->get_alloc(key);
        out_value = std::get<0>(r);
        out_value_len = std::get<1>(r);
      }
      catch ( const std::bad_alloc & )
      {
        return Component::IKVStore::E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
      }
    }
    return Component::IKVStore::S_OK;
  }
  catch ( const std::out_of_range & )
  {
    return Component::IKVStore::E_KEY_NOT_FOUND;
  }
}

auto hstore::get_direct(const pool_t pool,
                        const std::string & key,
                        void* out_value,
                        std::size_t& out_value_len,
                        Component::IKVStore::memory_handle_t) -> status_t
{
  const auto session = static_cast<const session_t *>(locate_session(pool));
  if ( ! session )
  {
    return Component::IKVStore::E_POOL_NOT_FOUND;
  }

  try
  {
    const auto buffer_size = out_value_len;
    out_value_len = session->get(key, out_value, buffer_size);
    if ( buffer_size < out_value_len )
    {
      return Component::IKVStore::E_INSUFFICIENT_BUFFER;
    }
    return Component::IKVStore::S_OK;
  }
  catch ( const std::out_of_range & )
  {
    return Component::IKVStore::E_KEY_NOT_FOUND;
  }
}

auto hstore::get_attribute(
  const pool_t pool,
  const Attribute attr,
  std::vector<uint64_t>& out_attr,
  const std::string* key) -> status_t
{
  const auto session = static_cast<const session_t *>(locate_session(pool));
  if ( ! session )
  {
    return Component::IKVStore::E_POOL_NOT_FOUND;
  }

  switch ( attr )
  {
  case VALUE_LEN:
    if ( ! key )
    {
      return Component::IKVStore::E_BAD_PARAM;
    }
    try
    {
      /* interface does not say what we do to the out_attr vector;
       * push_back is at least non-destructive.
       */
      out_attr.push_back(session->get_value_len(*key));
      return Component::IKVStore::S_OK;
    }
    catch ( const std::out_of_range & )
    {
      return Component::IKVStore::E_KEY_NOT_FOUND;
    }
    catch ( const std::bad_alloc & )
    {
      return Component::IKVStore::E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
    }
  default:
    ;
  }
  return Component::IKVStore::E_NOT_SUPPORTED;
}

auto hstore::set_attribute(
  const pool_t pool,
  const Attribute attr
  , const std::vector<uint64_t> & value
  , const std::string *) -> status_t
{
  const auto session = static_cast<const session_t *>(locate_session(pool));
  if ( ! session )
  {
    return Component::IKVStore::E_POOL_NOT_FOUND;
  }
  switch ( attr )
  {
  case AUTO_HASHTABLE_EXPANSION:
    if ( value.size() < 1 )
    {
      return Component::IKVStore::E_BAD_PARAM;
    }
    if ( value[0] == 0 )
    {
      return Component::IKVStore::E_NOT_SUPPORTED;
    }
    return Component::IKVStore::S_OK;
  default:
    ;
  }

  return Component::IKVStore::E_NOT_SUPPORTED;
}

auto hstore::lock(const pool_t pool,
                  const std::string &key,
                  lock_type_t type,
                  void *& out_value,
                  std::size_t & out_value_len) -> Component::IKVStore::key_t
{
  const auto session = static_cast<session_t *>(locate_session(pool));
  return
    session
    ? session->lock(key, type, out_value, out_value_len)
    : KEY_NONE
    ;
}


auto hstore::unlock(const pool_t pool,
                    Component::IKVStore::key_t key_) -> status_t
{
  const auto session = static_cast<session_t *>(locate_session(pool));
  return
    session
    ? session->unlock(key_)
    : Component::IKVStore::E_POOL_NOT_FOUND
    ;
}

auto hstore::erase(const pool_t pool,
                   const std::string &key
                   ) -> status_t
{
  const auto session = static_cast<session_t *>(locate_session(pool));
  return session
    ? ( session->erase(key) == 0 ? Component::IKVStore::E_KEY_NOT_FOUND : Component::IKVStore::S_OK )
    : Component::IKVStore::E_POOL_NOT_FOUND
    ;
}

std::size_t hstore::count(const pool_t pool)
{
  const auto session = static_cast<session_t *>(locate_session(pool));
  if ( ! session )
  {
    return Component::IKVStore::E_POOL_NOT_FOUND;
  }

  return session->count();
}

void hstore::debug(const pool_t, const unsigned cmd, const uint64_t arg)
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
                 > f_
                 ) -> status_t
{
  const auto session = static_cast<session_t *>(locate_session(pool));

  return session
    ? ( session->map(f_), Component::IKVStore::S_OK )
    : Component::IKVStore::E_POOL_NOT_FOUND
    ;
}

auto hstore::map_keys(
                 pool_t pool,
                 std::function
                 <
                   int(const std::string &key)
                 > f_
                 ) -> status_t
{
  const auto session = static_cast<session_t *>(locate_session(pool));

  return session
    ? ( session->map([&f_] (const std::string &key, const void *, std::size_t) -> int { f_(key); return 0; }), Component::IKVStore::S_OK )
    : Component::IKVStore::E_POOL_NOT_FOUND
    ;
}

auto hstore::free_memory(void * p) -> status_t
{
  scalable_free(p);
  return Component::IKVStore::S_OK;
}

auto hstore::atomic_update(
                           const pool_t pool
                           , const std::string& key
                           , const std::vector<IKVStore::Operation *> &op_vector
                           , const bool take_lock) -> status_t
try
{
  const auto update_method = take_lock ? &session_t::lock_and_atomic_update : &session_t::atomic_update;
  const auto session = static_cast<session_t *>(locate_session(pool));
    return
      session
      ? ( (session->*update_method)(key, op_vector), Component::IKVStore::S_OK )
      : Component::IKVStore::E_POOL_NOT_FOUND
      ;
}
catch ( const std::bad_alloc & )
{
  return Component::IKVStore::E_TOO_LARGE; /* would be E_NO_MEM, if it were in the interface */
}
catch ( const std::invalid_argument & )
{
  return Component::IKVStore::E_NOT_SUPPORTED;
}
catch ( const std::system_error & )
{
  return Component::IKVStore::E_FAIL;
}

