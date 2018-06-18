#include <fcntl.h>
#include <iostream>
#include <set>
#include <string>
#include <stdio.h>
#include <mutex>
#include <api/kvstore_itf.h>
#include <common/city.h>
#include <common/exceptions.h>
#include <boost/filesystem.hpp>
#include <sys/stat.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_set.h>

#include "file_store.h"

using namespace Component;

namespace fs=boost::filesystem;


struct Pool_handle
{
  fs::path     path;
  unsigned int flags;

  int put(const std::string& key,
          const void * value,
          const size_t value_len);
  
  int get(const std::string key,
          void*& out_value,
          size_t& out_value_len);
  
  int get_direct(const std::string key,
                 void* out_value,
                 size_t out_value_len);
  
  int erase(const std::string key);

};

std::mutex              _pool_sessions_lock;
std::set<Pool_handle *> _pool_sessions;

using lock_guard = std::lock_guard<std::mutex>;


int Pool_handle::put(const std::string& key,
                     const void * value,
                     const size_t value_len)
{
  std::string full_path = path.string() + "/" + key;
  if(fs::exists(full_path)) {
    PERR("key exists: (%s)", key.c_str());
    return IKVStore::E_KEY_EXISTS;
  }
  
  int fd = open(full_path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
  if(fd == -1) {
    assert(0);
    return E_FAIL;
  }
  ssize_t ws = write(fd, value, value_len);
  if(ws != value_len)
    throw General_exception("file write failed");

  close(fd);
  return S_OK;
}


int Pool_handle::get(const std::string key,
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

int Pool_handle::get_direct(const std::string key,
                            void* out_value,
                            size_t out_value_len)
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
  
  ssize_t rs = read(fd, out_value, out_value_len);
  if(rs != out_value_len)
    throw General_exception("file read failed");

  close(fd);
  return S_OK;
}



int Pool_handle::erase(const std::string key)
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


FileStore::FileStore(const std::string owner, const std::string name)
{
}

FileStore::~FileStore()
{
}
  

IKVStore::pool_t FileStore::create_pool(const std::string path,
                                        const std::string name,
                                        const size_t size,
                                        unsigned int flags,
                                        uint64_t args)
{
  if(!fs::exists(path))
    throw API_exception("path (%s) does not exist", path.c_str());

  fs::path p = path + "/" + name;
  if(!fs::create_directory(p))
    throw API_exception("filestore: failed to create directory (%s)", p.string().c_str());

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

IKVStore::pool_t FileStore::open_pool(const std::string path,
                                      const std::string name,
                                      unsigned int flags)
{
  fs::path p = path + "/" + name;
  if(!fs::exists(path))
    throw API_exception("path (%s) does not exist", path.c_str());

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

void FileStore::close_pool(pool_t pid)
{
  auto handle = reinterpret_cast<Pool_handle*>(pid);
  if(_pool_sessions.count(handle) != 1)
    throw API_exception("bad pool handle");

  {
     lock_guard g(_pool_sessions_lock);
    _pool_sessions.erase(handle);
  }
}

void FileStore::delete_pool(const pool_t pid)
{
  auto handle = reinterpret_cast<Pool_handle*>(pid);
  if(_pool_sessions.count(handle) != 1)
    throw API_exception("bad pool handle");

  boost::filesystem::remove_all(handle->path);
}

int FileStore::put(IKVStore::pool_t pid,
                   std::string key,
                   const void * value,
                   size_t value_len)
{
  auto handle = reinterpret_cast<Pool_handle*>(pid);
  if(_pool_sessions.count(handle) != 1)
    throw API_exception("bad pool handle");

  return handle->put(key, value, value_len);
}

int FileStore::get(const pool_t pid,
                   const std::string key,
                   void*& out_value,
                   size_t& out_value_len)
{
  auto handle = reinterpret_cast<Pool_handle*>(pid);
  if(_pool_sessions.count(handle) != 1)
    throw API_exception("bad pool handle");
    
  return handle->get(key, out_value, out_value_len);
}

int FileStore::get_direct(const pool_t pid,
                          const std::string key,
                          void* out_value,
                          size_t out_value_len)
{
  auto handle = reinterpret_cast<Pool_handle*>(pid);
  if(_pool_sessions.count(handle) != 1)
    throw API_exception("bad pool handle");
  
  return handle->get_direct(key, out_value, out_value_len);
}


int FileStore::erase(const pool_t pid,
                     const std::string key)
{
  auto handle = reinterpret_cast<Pool_handle*>(pid);
  return handle->erase(key);
}

size_t FileStore::count(const pool_t pool)
{
  assert(0);
  return 0; // not implemented
}

void FileStore::debug(const pool_t pool, unsigned cmd, uint64_t arg)
{
}


/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == FileStore_factory::component_id()) {
    return static_cast<void*>(new FileStore_factory());
  }
  else return NULL;
}


