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
#include <map>
#include <memory>
#include <unordered_map>
#include <string>
#include <stdio.h>
#include <city.h>
#include <common/exceptions.h>
#include <common/rand.h>
#include <common/utils.h>
#include <sys/stat.h>
#include <tbb/scalable_allocator.h>
#include <nupm/nupm.h>
#include <libpmem.h>

#include "dummy_store.h"

using namespace Component;
using namespace Common;

static unsigned Dummy_store_instances = 0;


Dummy_store::Dummy_store(const std::string& owner,
                         const std::string& name,
                         const std::string& dax_map) :
  _ddm(std::make_unique<Devdax_mgr>(dax_map, true))
{
  PLOG("Dummy_store: daxmap=%s", dax_map.c_str());
  //  std::move(mgr_)
  // std::make_unique<Devdax_manager>(dax_map, bool(std::getenv("DAX_RESET")))
  //  _ddm = std::make_unique<nupm::Devdax_manager>(dax_map, true);
}

Dummy_store::~Dummy_store()
{
}

std::unordered_map<uint64_t, std::pair<void *,size_t>> sessions;

IKVStore::pool_t Dummy_store::create_pool(const std::string& name,
                                          const size_t size,
                                          unsigned int flags,
                                          uint64_t args)
{
  const std::string& fullpath = name;
  PLOG("Dummy_store::create_pool (%s,size=%lu)", fullpath.c_str(), size);
    
  auto uuid = CityHash64(fullpath.c_str(), fullpath.length());
  void * p = _ddm->create_region(uuid, 0, size);
  _ddm->debug_dump(0);
  sessions[uuid] = std::make_pair(p,size);
  return uuid;
}

IKVStore::pool_t Dummy_store::open_pool(const std::string& name,
                                        unsigned int flags)
{
  return IKVStore::POOL_ERROR;
  // const std::string& fullpath = name;
  // PLOG("Dummy_store::open_pool (%s)", fullpath.c_str());
  
  // auto uuid = CityHash64(fullpath.c_str(), fullpath.length());
  // size_t len = 0;
  // void * p = _ddm->open_region(uuid, 0, &len);
  // _ddm->debug_dump(0);
  // sessions[uuid] = std::make_pair(p,size);
  // return uuid;  
}

status_t Dummy_store::close_pool(const pool_t pid)
{
  auto i = sessions.find(pid);
  if(i == sessions.end())
    return E_INVAL;
  sessions.erase(i);

  return S_OK;
}

status_t Dummy_store::delete_pool(const std::string& name)
{
  const std::string& fullpath = name;

  auto uuid = CityHash64(fullpath.c_str(), fullpath.length());
  for (auto& s : sessions) {
    if(s.first == uuid)
      return Component::IKVStore::E_ALREADY_OPEN;
  }
    
  _ddm->erase_region(uuid, 0);
  return S_OK;
}


status_t Dummy_store::put(IKVStore::pool_t pid,
                          const std::string& key,
                          const void * value,
                          size_t value_len,
                          unsigned int flags)
{
  auto i = sessions.find(pid);
  if(i == sessions.end())
    throw API_exception("delete_pool bad pool for Dummy_store");

  if(_debug_level > 1)
    PLOG("Dummy_store::put value_len=%lu", value_len);

  /* select some random location in the region */
  /* note:alignement makes a huge difference */
  uint64_t offset = round_down(genrand64_int64() % (GB(16) - value_len), 64);
  void * p = (void*) (((uint64_t)i->second.first) + offset);

  /* copy and flush */
  pmem_memcpy(p,
              value,
              value_len,
              PMEM_F_MEM_NONTEMPORAL | PMEM_F_MEM_WC );  

  return S_OK;
}


status_t Dummy_store::get(const pool_t pid,
                          const std::string& key,
                          void*& out_value,
                          size_t& out_value_len)
{
  auto i = sessions.find(pid);
  if(i == sessions.end())
    throw API_exception("delete_pool bad pool for Dummy_store");

  assert(out_value_len > 0);
  /* select some random location in the region */
  /* note:alignement makes a huge difference */
  uint64_t offset = round_down(genrand64_int64() % (i->second.second - 64), 64);
  void * p = (void*) (((char*)i->second.first) + offset);

  /* copy */
  out_value = malloc(64);
  out_value_len = 64;

  memcpy(out_value, p, out_value_len);
  return S_OK;
}

status_t Dummy_store::get_direct(const pool_t pid,
                               const std::string& key,
                               void* out_value,
                               size_t& out_value_len,
                               Component::IKVStore::memory_handle_t handle)
{
    auto i = sessions.find(pid);
  if(i == sessions.end())
    throw API_exception("delete_pool bad pool for Dummy_store");

  assert(out_value_len > 0);
  /* select some random location in the region */
  /* note:alignement makes a huge difference */
  uint64_t offset = round_down(genrand64_int64() % (i->second.second - 64), 64);
  void * p = (void*) (((char*)i->second.first) + offset);

  out_value_len = 64;

  memcpy(out_value, p, out_value_len);
  return S_OK;
}

status_t Dummy_store::put_direct(const pool_t pid,
                                 const std::string& key,
                                 const void * value,
                                 const size_t value_len,
                                 memory_handle_t memory_handle,
                                 unsigned int flags)
{
  return S_OK;
}

status_t Dummy_store::lock(const pool_t pid,
                  const std::string& key,
                  lock_type_t type,
                  void*& out_value,
                  size_t& out_value_len,
                  IKVStore::key_t &out_key)
{
  out_key = nullptr;
  return E_FAIL;
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
  return E_FAIL;
}

status_t Dummy_store::free_memory(void * p)
{
  ::free(p);
  return S_OK;
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


