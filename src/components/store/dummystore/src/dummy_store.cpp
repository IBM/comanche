#include <cerrno>
#include <fcntl.h>
#include <iostream>
#include <set>
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

#define SINGLE_THREADED
#include "dummy_store.h"

using namespace Component;
using namespace Common;


Dummy_store::Dummy_store(const std::string& owner, const std::string& name)
{
}

Dummy_store::~Dummy_store()
{
}
  

IKVStore::pool_t Dummy_store::create_pool(const std::string& path,
                                          const std::string& name,
                                          const size_t size,
                                          unsigned int flags,
                                          uint64_t args)
{
  std::string fullpath = path;
  fullpath += name;
  return CityHash64(fullpath.c_str(), fullpath.length());
}

IKVStore::pool_t Dummy_store::open_pool(const std::string& path,
                                      const std::string& name,
                                      unsigned int flags)
{
  std::string fullpath = path;
  fullpath += name;
  return CityHash64(fullpath.c_str(), fullpath.length());
}

void Dummy_store::close_pool(const pool_t pid)
{
}

void Dummy_store::delete_pool(const pool_t pid)
{
}


status_t Dummy_store::put(IKVStore::pool_t pid,
                        const std::string& key,
                        const void * value,
                        size_t value_len)
{
  return S_OK;
}

status_t Dummy_store::get(const pool_t pid,
                          const std::string& key,
                          void*& out_value,
                          size_t& out_value_len)
{
  out_value = malloc(32);
  memset(out_value, 'a', 32);
  out_value_len = 32;
  return S_OK;
}

status_t Dummy_store::get_direct(const pool_t pid,
                               const std::string& key,
                               void* out_value,
                               size_t& out_value_len,
                               Component::IKVStore::memory_handle_t handle)
{
  if(out_value_len < 32) return E_FAIL;
  memset(out_value, 'a', 32);
  out_value_len = 32;
  return S_OK;
}

status_t Dummy_store::put_direct(const pool_t pid,
                               const std::string& key,
                               const void * value,
                               const size_t value_len,
                               memory_handle_t memory_handle)
{
  return S_OK;
}

Component::IKVStore::key_t
Dummy_store::lock(const pool_t pid,
                  const std::string& key,
                  lock_type_t type,
                  void*& out_value,
                  size_t& out_value_len)
{
  return nullptr;
}

status_t Dummy_store::unlock(const pool_t pid,
                           key_t key_handle)
{
  return S_OK;
}

status_t Dummy_store::erase(const pool_t pid,
                          const std::string& key)
{
  return S_OK;
}

size_t Dummy_store::count(const pool_t pid)
{
  return 0;
}

void Dummy_store::free_memory(void * p)
{
  ::free(p);
}

void Dummy_store::debug(const pool_t pool, unsigned cmd, uint64_t arg)
{
}


/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Dummy_store_factory::component_id()) {
    return static_cast<void*>(new Dummy_store_factory());
  }
  else return NULL;
}


