#include <iostream>
#include <set>
#include <string>
#include <stdio.h>
#include <api/kvstore_itf.h>
#include <common/city.h>
#include <boost/filesystem.hpp>
#include <tbb/concurrent_hash_map.h>
#include "rocksdb/db.h"

#include "rocksdb_store.h"

using namespace Component;

RockStore::RockStore(const std::string owner, const std::string name)
{
}

RockStore::~RockStore()
{
}
  

IKVStore::pool_t RockStore::create_pool(const std::string path,
                                        const std::string name,
                                        const size_t size,
                                        unsigned int flags)
{
  rocksdb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.max_open_files = 1024;
  
  std::string db_file = path + "/" + name;
  rocksdb::DB* db = nullptr;
  rocksdb::Status status = rocksdb::DB::Open(options, db_file, &db);
  if(status.ok()==false)
    throw General_exception("unable to create rocksDB database (%s)", db_file.c_str());
  return reinterpret_cast<IKVStore::pool_t>(db);
}

IKVStore::pool_t RockStore::open_pool(const std::string path,
                                      const std::string name,
                                      unsigned int flags)
{
  rocksdb::Options options;
  options.create_if_missing = true;

  std::string db_file = path + "/" + name;
  rocksdb::DB* db = nullptr;
  rocksdb::Status status;
  if(flags & IKVStore::FLAGS_READ_ONLY)
    status = rocksdb::DB::OpenForReadOnly(options, db_file, &db);
  else
    status = rocksdb::DB::Open(options, db_file, &db);
  
  if(!status.ok())
    throw General_exception("unable to open or create rocksDB database (%s)", db_file.c_str());
  assert(db);
  return reinterpret_cast<IKVStore::pool_t>(db);
}

void RockStore::close_pool(pool_t pid)
{
  rocksdb::DB * db = reinterpret_cast<rocksdb::DB*>(pid);
  delete db;
}

int RockStore::put(IKVStore::pool_t pool,
                   std::string key,
                   const void * value,
                   size_t value_len)
{
  rocksdb::DB * db = reinterpret_cast<rocksdb::DB*>(pool);
  std::string val((const char *) value, value_len);
  
  rocksdb::WriteOptions write_options;
  write_options.disableWAL = true;
  // write_options.sync = true;
  
  rocksdb::Status status = db->Put(write_options, key, val);

  if(!status.ok()) {
    PERR("RockStore::put error (%s)", status.ToString().c_str());
    return E_FAIL;
  }

  if(option_SANITY)
  {
    PLOG("Put: key(%s)", key.c_str());      
    std::string value;
    rocksdb::Status status = db->Get(rocksdb::ReadOptions(), key, &value);
    if(!status.ok()) {
      PLOG("RockStore::get - sanity check failed (%s)", key.c_str());
      return E_FAIL;
    }
  }

    
  return S_OK;
}

int RockStore::get(const pool_t pool,
                    const std::string key,
                    void*& out_value,
                    size_t& out_value_len)
{
  rocksdb::DB * db = reinterpret_cast<rocksdb::DB*>(pool);
  std::string value;
  rocksdb::Status status = db->Get(rocksdb::ReadOptions(), key, &value);
  if(!status.ok()) {
    PLOG("RockStore::get - no key (%s)", key.c_str());
    return E_FAIL;
  }
  
  /* we have to do memcpy */
  out_value_len = value.length();
  out_value = malloc(out_value_len);
  assert(out_value);
  memcpy(out_value, value.data(), out_value_len);

  return S_OK;
}

int RockStore::erase(const pool_t pool,
                     const std::string key)
{
  rocksdb::DB * db = reinterpret_cast<rocksdb::DB*>(pool);
  rocksdb::Status status = db->Delete(rocksdb::WriteOptions(), key);
  if(!status.ok())
    return E_FAIL;

  return S_OK;
}

size_t RockStore::count(const pool_t pool)
{
  rocksdb::DB * db = reinterpret_cast<rocksdb::DB*>(pool);
  size_t num = 0;
  PWRN("RockStore::count is estimate only");
  std::string num_str;
  db->GetProperty("rocksdb.estimate-num-keys", &num_str);
  return std::strtoull(num_str.c_str(), NULL, 10);
}

void RockStore::debug(const pool_t pool, unsigned cmd, uint64_t arg)
{
}


/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == RockStore_factory::component_id()) {
    return static_cast<void*>(new RockStore_factory());
  }
  else return NULL;
}

#undef RESET_STATE
