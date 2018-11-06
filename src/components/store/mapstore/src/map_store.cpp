#include <cerrno>
#include <fcntl.h>
#include <iostream>
#include <set>
#include <map>
#include <string>
#include <stdio.h>
#include <api/kvstore_itf.h>
#include <common/city.h>
#include <common/rwlock.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <sys/stat.h>
#include <tbb/scalable_allocator.h>

#include "map_store.h"

using namespace Component;
using namespace Common;

static constexpr size_t OBJECT_ALIGNMENT = 4096;

static uint64_t get_key_hash(const std::string& key)
{
  return CityHash64(key.c_str(), key.length());
}

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
  std::string                       key;
  std::map<uint64_t, Value_pair>    map; /*< rb-tree based map */
  Common::RWLock                    map_lock; /*< read write lock */
  unsigned int                      flags;

  status_t put(const std::string& key,
               const void * value,
               const size_t value_len);
    
  status_t get(const std::string& key,
               void*& out_value,
               size_t& out_value_len);
  
  status_t get_direct(const std::string& key,
                      void* out_value,
                      size_t& out_value_len);

  status_t lock(uint64_t key_hash,
                IKVStore::lock_type_t type,
                void*& out_value,
                size_t& out_value_len);

  status_t unlock(uint64_t key_hash);

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

std::mutex                            _pool_sessions_lock;
std::set<Pool_session *>              _pool_sessions;
std::map<std::string, Pool_handle *>  _pools; /*< existing pools */

using Std_lock_guard = std::lock_guard<std::mutex>;

static Pool_session* get_session(const IKVStore::pool_t pid)
{
  auto session = reinterpret_cast<Pool_session*>(pid);

  if(_pool_sessions.count(session) != 1)
    throw API_exception("invalid pool identifier");

  assert(session);
  return session;
}

status_t Pool_handle::put(const std::string& key,
                          const void * value,
                          const size_t value_len)
{
  if(!value || !value_len)
    throw API_exception("invalid parameters");

  uint64_t hashkey = get_key_hash(key);

  RWLock_guard guard(map_lock, RWLock_guard::WRITE);
  
  if(map.find(hashkey) != map.end()) {
    // if(option_DEBUG)
    //   PLOG("Map_store:: key already exists (%s)", key.c_str());
    
    return IKVStore::E_KEY_EXISTS;
  }

  auto buffer = scalable_aligned_malloc(value_len, OBJECT_ALIGNMENT);
  memcpy(buffer, value, value_len);
  map.emplace(hashkey, Value_pair{buffer, value_len});
  scalable_free(buffer);

  return S_OK;
}


status_t Pool_handle::get(const std::string& key,
                          void*& out_value,
                          size_t& out_value_len)
{
  RWLock_guard guard(map_lock);

  auto i = map.find(get_key_hash(key));

  if(i == map.end()) {
    return IKVStore::E_KEY_NOT_FOUND;
  }

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

  RWLock_guard guard(map_lock);
  auto i = map.find(get_key_hash(key));

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

status_t Pool_handle::lock(uint64_t key_hash,
                           IKVStore::lock_type_t type,
                           void*& out_value,
                           size_t& out_value_len)
{
  void * buffer = nullptr;
  
  /* on-demand create */
  {
    RWLock_guard guard(map_lock, RWLock_guard::WRITE);

    auto i = map.find(key_hash);
    
    if(i == map.end()) {
      buffer = scalable_aligned_malloc(out_value_len, OBJECT_ALIGNMENT);
      assert(buffer);
      map.emplace(key_hash, Value_pair{buffer, out_value_len});
    }   
  }
  
  if(type == IKVStore::STORE_LOCK_READ)
    map_lock.read_lock();
  else if(type == IKVStore::STORE_LOCK_WRITE)
    map_lock.write_lock();
  else throw API_exception("invalid lock type");

  if(!buffer) {
    auto entry = map.find(key_hash);
    assert(entry != map.end());
    buffer = entry->second.ptr;
    out_value_len = entry->second.length;
  }

  out_value = buffer;
      
  return S_OK;
}

status_t Pool_handle::unlock(uint64_t key_hash)
{
  /* this component only supports pool level locking
     so key_hash is not needed really */
  auto i = map.find(key_hash);
  
  if(i == map.end())
    return IKVStore::E_KEY_NOT_FOUND;

  map_lock.unlock();
  return S_OK;
}

status_t Pool_handle::erase(const std::string& key)
{
  RWLock_guard guard(map_lock, RWLock_guard::WRITE);

  auto i = map.find(get_key_hash(key));

  if(i == map.end())
    return IKVStore::E_KEY_NOT_FOUND;

  map.erase(i);
  scalable_free(i->second.ptr);

  return S_OK;
}

size_t Pool_handle::count() {
  RWLock_guard guard(map_lock);
  return map.size();
}

/** Main class */

Map_store::Map_store(const std::string& owner, const std::string& name)
{
}

Map_store::~Map_store()
{
  Std_lock_guard g(_pool_sessions_lock);
  for(auto& s : _pool_sessions)
    delete s;
    
  for(auto& p : _pools)
    delete p.second;
}
  

IKVStore::pool_t Map_store::create_pool(const std::string& path,
                                        const std::string& name,
                                        const size_t size,
                                        unsigned int flags,
                                        uint64_t args)
{
  if(flags & FLAGS_READ_ONLY)
    throw API_exception("read only create_pool not supported on map-store component");

  const auto handle = new Pool_handle;
  Pool_session * session = nullptr;
  handle->key = path + "/" + name;
  handle->flags = flags;
  {
    Std_lock_guard g(_pool_sessions_lock);

    if(flags & FLAGS_CREATE_ONLY) {
      if(_pools.find(handle->key) != _pools.end()) {
        delete handle;
        throw General_exception("pool already exists");
      }
    }
    session = new Pool_session{handle};
    _pools[handle->key] = handle;
    _pool_sessions.insert(session); /* create a session too */
  }
  
  if(option_DEBUG)
    PLOG("Map_store: created pool OK: %s", handle->key.c_str());

  assert(session);
  return reinterpret_cast<IKVStore::pool_t>(session);
}

IKVStore::pool_t Map_store::open_pool(const std::string& path,
                                      const std::string& name,
                                      unsigned int flags)
{
  std::string key = path + name;

  Pool_handle * ph = nullptr;
  /* see if a pool exists that matches the key */
  for(auto& h: _pools) {
    if(h.first == key) {
      ph = h.second;
      break;
    }
  }

  if(ph == nullptr)
    throw API_exception("open_pool failed; pool (%s) does not exist", key.c_str());

  auto new_session = new Pool_session(ph);
  if(option_DEBUG)
    PLOG("opened pool(%p)", new_session);
  _pool_sessions.insert(new_session);
  
  return reinterpret_cast<IKVStore::pool_t>(new_session);
}

void Map_store::close_pool(const pool_t pid)
{
  if(option_DEBUG)
    PLOG("close_pool(%p)", (void*) pid);
  
  auto session = get_session(pid);

  Std_lock_guard g(_pool_sessions_lock);
  _pool_sessions.erase(session);
}

void Map_store::delete_pool(const pool_t pid)
{
  auto session = get_session(pid);

  Std_lock_guard g(_pool_sessions_lock);
  _pool_sessions.erase(session);

  /* delete pool too */
  for(auto& p : _pools) {
    if(p.second == session->pool) {
      _pools.erase(p.first);
      return;
    }
  }
  throw Logic_exception("unable to find pool to delete");
}


status_t Map_store::put(IKVStore::pool_t pid,
                        const std::string& key,
                        const void * value,
                        size_t value_len)
{
  auto session = get_session(pid);
  assert(session->pool);
  return session->pool->put(key, value, value_len);  
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
                               memory_handle_t memory_handle)
{
  auto session = get_session(pid);
  return Map_store::put(pid, key, value, value_len);
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

  auto hash = get_key_hash(key);
  session->pool->lock(hash, type, out_value, out_value_len);
  return reinterpret_cast<key_t>(hash);
}

status_t Map_store::unlock(const pool_t pid,
                           key_t key_handle)
{
  auto session = get_session(pid);
  assert(session->pool);

  return session->pool->unlock(reinterpret_cast<uint64_t>(key_handle));
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

void Map_store::free_memory(void * p)
{
  scalable_free(p);
}

void Map_store::debug(const pool_t pool, unsigned cmd, uint64_t arg)
{
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


