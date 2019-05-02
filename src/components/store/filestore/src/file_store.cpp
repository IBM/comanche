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
#include <string>
#include <stdio.h>
#include <mutex>
#include <api/kvstore_itf.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <sys/stat.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_set.h>

#include "file_store.h"

//#define FORCE_FLUSH  // enable this for GPU testing
using namespace Component;

namespace fs=boost::filesystem;

class Simulated_locked_item
{
public:
  Simulated_locked_item(size_t size) {
    p = ::malloc(size);
    p_len = size;
  }
  ~Simulated_locked_item() {
    ::free(p);
  }
  void * p;
  size_t p_len;
};

struct Pool_handle
{
  fs::path     path;
  unsigned int flags;
  int use_cache = 1;

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

  status_t erase(const std::string& key);

  status_t get_attribute(const IKVStore::Attribute attr,
                         std::vector<uint64_t>& out_value,
                         const std::string* key);

  IKVStore::key_t lock(const std::string& key,
                       IKVStore::lock_type_t type,
                       void*& out_value,
                       size_t& out_value_len);

  status_t unlock(IKVStore::key_t key_handle);

  size_t count() const;

};

std::mutex              _pool_sessions_lock;
std::set<Pool_handle *> _pool_sessions;

using lock_guard = std::lock_guard<std::mutex>;


status_t Pool_handle::put(const std::string& key,
                          const void * value,
                          const size_t value_len,
                          unsigned int flags)
{
  std::string full_path = path.string() + "/" + key;

  if(fs::exists(full_path)) {

    if(flags & IKVStore::FLAGS_DONT_STOMP) {
      return IKVStore::E_KEY_EXISTS;
    }
    else {
      erase(key);
    }
  }
  
  int fd = open(full_path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
  if(fd == -1) {
    //assert(0);
    std::perror("open in put call returned -1");
    return E_FAIL;
  }
  
  ssize_t ws = write(fd, value, value_len);
  if(ws != value_len)
    throw General_exception("file write failed, value=%p, len =%lu", value, value_len);

  /*Turn on to avoid the effect of file cache*/
  if (!use_cache)
    {
      fsync(fd);
    }

  close(fd);
  return S_OK;
}


status_t Pool_handle::get(const std::string& key,
                          void*& out_value,
                          size_t& out_value_len)
{
  PLOG("get: key=(%s) path=(%s)", key.c_str(), path.string().c_str());
  
  std::string full_path = path.string() + "/" + key;
  if(!fs::exists(full_path)) {
    PERR("key not found: (%s)", full_path.c_str());
    return IKVStore::E_KEY_NOT_FOUND;
  }

  int fd = open(full_path.c_str(), O_RDONLY, 0644);
  
  struct stat buffer;
  if(stat(full_path.c_str(), &buffer))
    throw General_exception("stat failed on file (%s)", full_path.c_str());

  assert(buffer.st_size > 0);
  out_value = malloc(buffer.st_size);
  out_value_len = buffer.st_size;

  ssize_t rs = read(fd, out_value, out_value_len);
  if(rs != out_value_len)
    throw General_exception("file read failed");

  close(fd);
  return S_OK;
}

status_t Pool_handle::get_direct(const std::string& key,
                                 void* out_value,
                                 size_t& out_value_len)
{
  PLOG("get: key=(%s) path=(%s)", key.c_str(), path.string().c_str());
  
  std::string full_path = path.string() + "/" + key;
  if(!fs::exists(full_path)) {
    PERR("key not found: (%s)", full_path.c_str());
    return IKVStore::E_KEY_NOT_FOUND;
  }

  int fd = open(full_path.c_str(), O_RDONLY, 0644);
  
  struct stat buffer;
  if(stat(full_path.c_str(), &buffer))
    throw General_exception("stat failed on file (%s)", full_path.c_str());

  if(out_value_len < buffer.st_size)
    return IKVStore::E_INSUFFICIENT_BUFFER;

  out_value_len = buffer.st_size;
  
  ssize_t rs = read(fd, out_value, out_value_len);

  if(rs != out_value_len)
    {
      perror("get_direct read size didn't match expected out_value_len");
      throw General_exception("file read failed");
    }

#ifdef FORCE_FLUSH
  clflush_area(out_value, out_value_len);
#endif

  close(fd);
  return S_OK;
}

IKVStore::key_t Pool_handle::lock(const std::string& key,
                                  IKVStore::lock_type_t type,
                                  void*& out_value,
                                  size_t& out_value_len)
{
  std::string full_path = path.string() + "/" + key;
  if(!fs::exists(full_path))
    return IKVStore::KEY_NONE;

  struct stat info;
  if(stat(full_path.c_str(), &info))
    throw General_exception("stat failed on file (%s)", full_path.c_str());

  auto sl = new Simulated_locked_item(info.st_size);
  assert(sl);
  
  /* this component doesn't really lock, we simulate it though */
  int fd = ::open(full_path.c_str(), O_RDONLY, 0644);
  out_value_len = info.st_size;
  out_value = sl->p;
  ssize_t rs = read(fd, out_value, out_value_len);

  if(rs != out_value_len)
    throw General_exception("file read failed (rs=%lu)", rs);

  ::close(fd);

  return reinterpret_cast<IKVStore::key_t>(sl);
}

status_t Pool_handle::unlock(IKVStore::key_t key_handle)
{
  Simulated_locked_item * li = reinterpret_cast<Simulated_locked_item*>(key_handle);  
  delete li;
}



status_t Pool_handle::erase(const std::string& key)
{
  std::string full_path = path.string() + "/" + key;
  if(!fs::exists(full_path))
    return IKVStore::E_KEY_NOT_FOUND;

  if(!fs::remove(full_path)) {
    assert(0);
    return E_FAIL;
  }

  return S_OK;
}

status_t Pool_handle::get_attribute(const IKVStore::Attribute attr,
                                    std::vector<uint64_t>& out_value,
                                    const std::string* key)
{
  switch(attr)
    {
    case IKVStore::Attribute::VALUE_LEN:
      {
        if(key == nullptr) return E_FAIL;
        
        std::string full_path = path.string() + "/" + *key;
      
        struct stat info;
        if(stat(full_path.c_str(), &info))
          throw General_exception("stat failed on file (%s)", full_path.c_str());
        out_value.push_back(info.st_size);
        return S_OK;
      }
    case IKVStore::Attribute::COUNT:
      {
        using namespace boost::filesystem;
        std::string dir = path.string() + "/";
        fs::path p(dir);
        if(is_directory(dir)) {
          size_t count = 0;
          for(auto& entry : boost::make_iterator_range(directory_iterator(p), {}))
            count ++;
          out_value.push_back(count);
          return S_OK;
        }
        else throw Logic_exception("iterating dir");
        return S_OK;
      }
    default:
      return IKVStore::E_BAD_PARAM;
    }
  
  return E_NOT_IMPL;
}

size_t Pool_handle::count() const
{
  using namespace boost::filesystem;
  std::string dir = path.string() + "/";
  fs::path p(dir);
  if(is_directory(dir)) {
    size_t count = 0;
    for(auto& entry : boost::make_iterator_range(directory_iterator(p), {}))
      count ++;
    return count;
  }
  else throw Logic_exception("not dir");
}

File_store::File_store(const std::string& path) : _root_path(path)
{
  if(_root_path.back() != '/')
    _root_path += "/";
}

File_store::~File_store()
{
}

IKVStore::key_t File_store::lock(const pool_t pool,
                                 const std::string& key,
                                 IKVStore::lock_type_t type,
                                 void*& out_value,
                                 size_t& out_value_len)
{
  auto handle = reinterpret_cast<Pool_handle*>(pool);
  if(_pool_sessions.count(handle) != 1)
    return KEY_NONE;
  return handle->lock(key,type,out_value,out_value_len);
}

status_t File_store::unlock(const pool_t pool,
                            IKVStore::key_t key_handle)
{
  auto handle = reinterpret_cast<Pool_handle*>(pool);
  if(_pool_sessions.count(handle) != 1)
    return E_POOL_NOT_FOUND;
  return handle->unlock(key_handle);
}


int File_store::get_capability(Capability cap) const
{
  switch(cap) {
  case Capability::POOL_DELETE_CHECK: return 1;
  case Capability::POOL_THREAD_SAFE: return 1;
  case Capability::RWLOCK_PER_POOL: return 1;
  default: return -1;
  }
}

IKVStore::pool_t File_store::create_pool(const std::string& name,
                                         const size_t size,
                                         unsigned int flags,
                                         uint64_t args)
{
  fs::path p = _root_path + name;

  if(fs::exists(p)) {
    if(flags & IKVStore::FLAGS_CREATE_ONLY) return POOL_ERROR;
  }
  else if(!fs::create_directory(p)) {
    throw General_exception("filestore unable to create dir (%s)", p.string().c_str());
  }

  if(option_DEBUG)
    PLOG("created pool OK: %s", p.string().c_str());

  auto handle = new Pool_handle;
  handle->path = p;
  handle->flags = flags;
  {
    lock_guard g(_pool_sessions_lock);
    _pool_sessions.insert(handle);
  }
  return reinterpret_cast<IKVStore::pool_t>(handle);
}

IKVStore::pool_t File_store::open_pool(const std::string& name,
                                       unsigned int flags)
{
  fs::path p = _root_path + name;
  if(!fs::exists(name))
    return POOL_ERROR;

  if(option_DEBUG)
    PLOG("opened pool OK: %s", p.string().c_str());
  
  auto handle = new Pool_handle;
  handle->path = p;

  {
    lock_guard g(_pool_sessions_lock);
    _pool_sessions.insert(handle);
  }
  
  return reinterpret_cast<IKVStore::pool_t>(handle);
}

status_t File_store::close_pool(pool_t pool)
{
  auto handle = reinterpret_cast<Pool_handle*>(pool);
  if(_pool_sessions.count(handle) != 1)
    return E_INVAL;

  {
    lock_guard g(_pool_sessions_lock);
    _pool_sessions.erase(handle);
  }
  return S_OK;
}

status_t File_store::delete_pool(const std::string &name)
{
  std::string fullpath = _root_path + name;
  if(!fs::exists(fullpath))
    return E_POOL_NOT_FOUND;

  lock_guard g(_pool_sessions_lock);
  for(auto& s: _pool_sessions) {
    if(s->path == fullpath)
      return E_ALREADY_OPEN;
  }
    
  boost::filesystem::remove_all(fullpath);
  return S_OK;
}

status_t File_store::put(IKVStore::pool_t pool,
                         const std::string& key,
                         const void * value,
                         size_t value_len,
                         unsigned int flags)
{
  auto handle = reinterpret_cast<Pool_handle*>(pool);
  if(_pool_sessions.count(handle) != 1)
    throw API_exception("bad pool handle");

  return handle->put(key, value, value_len, flags);
}

status_t File_store::put_direct(const pool_t pool,
                                const std::string& key,
                                const void * value,
                                const size_t value_len,
                                memory_handle_t memory_handle,
                                unsigned int flags)
{
  return File_store::put(pool, key, value, value_len, flags);
}


status_t File_store::get(const pool_t pool,
                         const std::string& key,
                         void*& out_value,
                         size_t& out_value_len)
{
  auto handle = reinterpret_cast<Pool_handle*>(pool);
  if(_pool_sessions.count(handle) != 1)
    return E_POOL_NOT_FOUND;
    
  return handle->get(key, out_value, out_value_len);
}

status_t File_store::get_direct(const pool_t pool,
                                const std::string& key,
                                void* out_value,
                                size_t& out_value_len,
                                Component::IKVStore::memory_handle_t handle)
{
  auto pool_handle = reinterpret_cast<Pool_handle*>(pool);
  if(_pool_sessions.count(pool_handle) != 1)
    return E_POOL_NOT_FOUND;
  
  return pool_handle->get_direct(key, out_value, out_value_len);
}

status_t File_store::erase(const pool_t pool,
                           const std::string& key)
{
  auto handle = reinterpret_cast<Pool_handle*>(pool);
  return handle->erase(key);
}

size_t File_store::count(const pool_t pool)
{
  auto pool_handle = reinterpret_cast<Pool_handle*>(pool);
  if(_pool_sessions.count(pool_handle) != 1)
    throw API_exception("bad pool handle");
    
  return pool_handle->count();
}

void File_store::debug(const pool_t pool, unsigned cmd, uint64_t arg)
{
}

status_t File_store::get_attribute(const pool_t pool,
                                   const IKVStore::Attribute attr,
                                   std::vector<uint64_t>& out_value,
                                   const std::string* key)
{
  if(attr == IKVStore::Attribute::VALUE_LEN) {
    auto handle = reinterpret_cast<Pool_handle*>(pool);
    return handle->get_attribute(attr, out_value, key);
  }
  return E_NOT_IMPL;
}


/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == File_store_factory::component_id()) {
    return static_cast<void*>(new File_store_factory());
  }
  else return NULL;
}


