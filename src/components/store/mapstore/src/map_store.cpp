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
#include <cerrno>
#include <fcntl.h>
#include <iostream>
#include <set>
#include <unordered_map>
#include <map>
#include <string>
#include <stdio.h>
#include <api/kvstore_itf.h>
#include <city.h>
#include <common/rwlock.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <sys/stat.h>
#include <tbb/scalable_allocator.h>

#define OBJECT_ALIGNMENT 8
#define SINGLE_THREADED
#include "map_store.h"

using namespace Component;
using namespace Common;

template<typename X, typename Y>
using map_t = std::unordered_map<X,Y>;


struct Value_pair
{
  void * ptr;
  size_t length;
};

class Pool_handle
{
private:
  static constexpr bool option_DEBUG = false;

public:
  std::string                       name;
  map_t<std::string, Value_pair>    map; /*< hash table based map */
  Common::RWLock                    map_lock; /*< read write lock */
  unsigned int                      flags;

  status_t put(const std::string& key,
               const void * value,
               const size_t value_len,
               unsigned int flags);
    
  status_t get(const std::string& key,
               void*& out_value,
               size_t& out_value_len);
  
  status_t get_direct(const std::string& key,
                      void* out_value,
                      size_t& out_value_len);

  IKVStore::key_t lock(const std::string& key,
                       IKVStore::lock_type_t type,
                       void*& out_value,
                       size_t& out_value_len);

  void unlock();

  status_t erase(const std::string& key);

  size_t count();
};

struct Pool_session
{
  Pool_session(Pool_handle * ph) : pool(ph) {}
  bool check() const { return canary == 0x45450101; }
  Pool_handle * pool;
  const unsigned canary = 0x45450101;
};

struct tls_cache_t {
  Pool_session * session;
};

std::mutex                            _pool_sessions_lock;
std::set<Pool_session *>              _pool_sessions;
std::unordered_map<std::string, Pool_handle *>  _pools; /*< existing pools */
static __thread tls_cache_t tls_cache = { nullptr };

using Std_lock_guard = std::lock_guard<std::mutex>;

Pool_session* get_session(const IKVStore::pool_t pid)
{
  auto session = reinterpret_cast<Pool_session*>(pid);

  if ( session != tls_cache.session)
  {
    Std_lock_guard g(_pool_sessions_lock);

    if(_pool_sessions.count(session) == 0)
      throw API_exception("invalid pool identifier (%p)", pid);
    tls_cache.session = session;
  }

  assert(session);
  return session;
}

status_t Pool_handle::put(const std::string& key,
                          const void * value,
                          const size_t value_len,
                          unsigned int flags)
{
  if(!value || !value_len)
    throw API_exception("invalid parameters");

#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock, RWLock_guard::WRITE);
#endif

  auto i = map.find(key);
  if(i != map.end()) {
    if(flags & IKVStore::FLAGS_DONT_STOMP) {
      PWRN("put refuses to stomp (%s)", key.c_str());
      return IKVStore::E_KEY_EXISTS;
    }

    auto& p = i->second;
    if(p.length == value_len) {
      memcpy(p.ptr, value, value_len);
    }
    else {
      /* different size, reallocate */
      scalable_free(p.ptr);
      p.ptr = scalable_aligned_malloc(value_len, OBJECT_ALIGNMENT);
      memcpy(p.ptr, value, value_len);
    }
  }
  else {
    auto buffer = scalable_aligned_malloc(value_len, OBJECT_ALIGNMENT);
    memcpy(buffer, value, value_len);
    map[key] = Value_pair{buffer, value_len};
  }
  
  return S_OK;
}


status_t Pool_handle::get(const std::string& key,
                          void*& out_value,
                          size_t& out_value_len)
{
  if(option_DEBUG)
    PLOG("map_store: get(%s,%p,%lu)", key.c_str(), out_value, out_value_len);

#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock);
#endif
  
  auto i = map.find(key);

  if(i == map.end())
    return IKVStore::E_KEY_NOT_FOUND;

  out_value_len = i->second.length;
  out_value = scalable_aligned_malloc(out_value_len, OBJECT_ALIGNMENT);
  memcpy(out_value, i->second.ptr, i->second.length);
  
  return S_OK;
}

status_t Pool_handle::get_direct(const std::string& key,
                                 void* out_value,
                                 size_t& out_value_len)
{
  if(option_DEBUG)
    PLOG("Map_store GET: key=(%s) ", key.c_str());
  
  if(out_value == nullptr || out_value_len == 0)
    throw API_exception("invalid parameter");

#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock);
#endif
  auto i = map.find(key);

  if(i == map.end()) {
    if(option_DEBUG)
      PERR("Map_store: error key not found");
    return IKVStore::E_KEY_NOT_FOUND;
  }
  
  if(out_value_len < i->second.length) {
    if(option_DEBUG)
      PERR("Map_store: error insufficient buffer");

    return IKVStore::E_INSUFFICIENT_BUFFER;
  }

  out_value_len = i->second.length; /* update length */
  memcpy(out_value, i->second.ptr, i->second.length);
  
  return S_OK;
}

IKVStore::key_t Pool_handle::lock(const std::string& key,
                                  IKVStore::lock_type_t type,
                                  void*& out_value,
                                  size_t& out_value_len)
{
  void * buffer = nullptr;
  
  /* on-demand create */
  {
    RWLock_guard guard(map_lock, RWLock_guard::WRITE);

    auto i = map.find(key);
    //    PLOG("looking for key:(%s)%lu", key.c_str(),key.length());
    
    if(i == map.end()) {

      if(out_value_len == 0) {
        for(auto& i : map) {
           PLOG("key:(%s) length=%lu", i.first.c_str(), i.first.length());
         }
        throw General_exception("mapstore: tried to lock object that was not found and object size to create not given (key=%s)", key.c_str());
      }

      buffer = scalable_aligned_malloc(out_value_len, OBJECT_ALIGNMENT);

      if(buffer == nullptr)
        throw General_exception("Pool_handle::lock on-demand create scalable_aligned_malloc failed (len=%lu)",
                                out_value_len);
        
      assert(buffer);
      PLOG("lock emplacing key=(%s)", key.c_str());
      map.emplace(key, Value_pair{buffer, out_value_len});
    }   
  }
  
  if(type == IKVStore::STORE_LOCK_READ)
    map_lock.read_lock();
  else if(type == IKVStore::STORE_LOCK_WRITE)
    map_lock.write_lock();
  else throw API_exception("invalid lock type");

  if(!buffer) {
    auto entry = map.find(key);
    assert(entry != map.end());
    buffer = entry->second.ptr;
    out_value_len = entry->second.length;
  }

  out_value = buffer;

  return (IKVStore::key_t) buffer; 
}

void Pool_handle::unlock()
{
  map_lock.unlock();
}

status_t Pool_handle::erase(const std::string& key)
{
#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock, RWLock_guard::WRITE);
#endif
  auto i = map.find(key);

  if(i == map.end())
    return IKVStore::E_KEY_NOT_FOUND;

  map.erase(i);
  scalable_free(i->second.ptr);

  return S_OK;
}

size_t Pool_handle::count() {
#ifndef SINGLE_THREADED
  RWLock_guard guard(map_lock);
#endif
  return map.size();
}

/** Main class */

Map_store::Map_store(const std::string& owner, const std::string& name)
{
}

Map_store::~Map_store()
{
  // TODO: seg faults?
#if 0
  Std_lock_guard g(_pool_sessions_lock);

  for(auto& s : _pool_sessions)
    delete s;

  for(auto& p : _pools)
    delete p.second;
#endif
}
  

IKVStore::pool_t Map_store::create_pool(const std::string& name,
                                        const size_t size,
                                        unsigned int flags,
                                        uint64_t args)
{
  if(flags & IKVStore::FLAGS_READ_ONLY)
    throw API_exception("read only create_pool not supported on map-store component");

  const auto handle = new Pool_handle;
  Pool_session * session = nullptr;
  handle->name = name;
  handle->flags = flags;
  {
    Std_lock_guard g(_pool_sessions_lock);

    if(flags & IKVStore::FLAGS_CREATE_ONLY) {
      if(_pools.find(handle->name) != _pools.end()) {
        delete handle;
        return POOL_ERROR;
      }
    }
    session = new Pool_session{handle};
    _pools[handle->name] = handle;
    PLOG("map_store: adding new session (%p)", session);
    _pool_sessions.insert(session); /* create a session too */
  }
  
  if(option_DEBUG)
    PLOG("map_store: created pool OK: %s", handle->name.c_str());

  assert(session);
  return reinterpret_cast<IKVStore::pool_t>(session);
}

IKVStore::pool_t Map_store::open_pool(const std::string& name,
                                      unsigned int flags)
{
  const std::string& key = name;

  Pool_handle * ph = nullptr;
  /* see if a pool exists that matches the key */
  for(auto& h: _pools) {
    PLOG("pool:%s", h.first.c_str());
    if(h.first == key) {
      ph = h.second;
      break;
    }
  }

  if(ph == nullptr)
    return Component::IKVStore::POOL_ERROR;

  auto new_session = new Pool_session(ph);
  if(option_DEBUG)
    PLOG("map_store: opened pool(%p)", new_session);
  _pool_sessions.insert(new_session);
  
  return reinterpret_cast<IKVStore::pool_t>(new_session);
}

status_t Map_store::close_pool(const pool_t pid)
{
  if(option_DEBUG)
    PLOG("map_store: close_pool(%p)", (void*) pid);
  
  auto session = get_session(pid);
  assert(session);

  tls_cache.session = nullptr;
  Std_lock_guard g(_pool_sessions_lock);
  delete session;
  _pool_sessions.erase(session);
  if(option_DEBUG)
    PLOG("map_store: erased session %p", session);

  return S_OK;
}

status_t Map_store::delete_pool(const std::string& poolname)
{
  Std_lock_guard g(_pool_sessions_lock);

  return S_OK;
  Pool_handle * ph = nullptr;
  /* see if a pool exists that matches the poolname */
  for(auto& h: _pools) {
    if(h.first == poolname) {
      ph = h.second;
      break;
    }
  }

  if(ph == nullptr) {
    PWRN("map_store: delete_pool (%s) pool not found", poolname.c_str());
    return E_POOL_NOT_FOUND;
  }

  for(auto& s: _pool_sessions) {
    if(s->pool->name == poolname) {
      PWRN("map_store: delete_pool (%s) pool delete failed because pool still open (%p)", poolname.c_str(), s);
      return E_ALREADY_OPEN;
    }
  }
    
  /* delete pool too */
  if(_pools.find(poolname) == _pools.end())
    throw Logic_exception("unable to delete pool session");
  
  _pools.erase(poolname);
  delete ph;
  return S_OK;
}


status_t Map_store::put(IKVStore::pool_t pid,
                        const std::string& key,
                        const void * value,
                        size_t value_len,
                        unsigned int flags)
{
  auto session = get_session(pid);
  assert(session->pool);
  return session->pool->put(key, value, value_len, flags);  
}

status_t Map_store::get(const pool_t pid,
                        const std::string& key,
                        void*& out_value,
                        size_t& out_value_len)
{
  auto session = get_session(pid);
  assert(session->pool);
  return session->pool->get(key, out_value, out_value_len);
}

status_t Map_store::get_direct(const pool_t pid,
                               const std::string& key,
                               void* out_value,
                               size_t& out_value_len,
                               Component::IKVStore::memory_handle_t handle)
{
  auto session = get_session(pid);
  assert(session->pool);
  return session->pool->get_direct(key, out_value, out_value_len);
}

status_t Map_store::put_direct(const pool_t pid,
                               const std::string& key,
                               const void * value,
                               const size_t value_len,
                               memory_handle_t memory_handle,
                               unsigned int flags)
{
  return Map_store::put(pid, key, value, value_len, flags);
}

Component::IKVStore::key_t
Map_store::lock(const pool_t pid,
                const std::string& key,
                lock_type_t type,
                void*& out_value,
                size_t& out_value_len)
{
  auto session = get_session(pid);
  assert(session->pool);

  if(option_DEBUG)
    PLOG("map_store: lock(%s)", key.c_str());

  return session->pool->lock(key, type, out_value, out_value_len);
}

status_t Map_store::unlock(const pool_t pid,
                           key_t key_handle)
{
  auto session = get_session(pid);
  assert(session);
  assert(session->pool);

  session->pool->unlock();
  return S_OK;
}

status_t Map_store::erase(const pool_t pid,
                          const std::string& key)
{
  auto session = get_session(pid);
  assert(session->pool);
  return session->pool->erase(key);
}

size_t Map_store::count(const pool_t pid)
{
  auto session = get_session(pid);
  assert(session->pool);
  return session->pool->count();
}

status_t Map_store::free_memory(void * p)
{
  scalable_free(p);
  return S_OK;
}

void Map_store::debug(const pool_t pool, unsigned cmd, uint64_t arg)
{
}

int Map_store::get_capability(Capability cap) const
{
  switch(cap) {
  case Capability::POOL_DELETE_CHECK: return 1;
  case Capability::POOL_THREAD_SAFE: return 1;
  case Capability::RWLOCK_PER_POOL: return 1;
  default: return -1;
  }
}

/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Map_store_factory::component_id()) {
    return static_cast<void*>(new Map_store_factory());
  }
  else return NULL;
}


