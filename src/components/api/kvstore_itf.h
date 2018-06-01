/*
   Copyright [2017] [IBM Corporation]

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


#ifndef __API_KVSTORE_ITF__
#define __API_KVSTORE_ITF__

#include <functional>

#include <api/components.h>
#include <api/block_itf.h>
#include <api/block_allocator_itf.h>

namespace Component
{

/** 
 * Key-value interface
 */
class IKVStore : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0x62f4829f,0x0405,0x4c19,0x9898,0xa3,0xae,0x21,0x5a,0x3e,0xe8);

public:
  using pool_t     = uint64_t;

  enum {
    THREAD_MODEL_UNSAFE,
    THREAD_MODEL_SINGLE_PER_POOL,
    THREAD_MODEL_MULTI_PER_POOL,
  };

  enum {
    FLAGS_READ_ONLY = 1,
  };

  enum {
    STORE_LOCK_READ=1,
    STORE_LOCK_WRITE=2,
  };

  enum {
    S_OK = 0,
    E_FAIL = -1,
    E_KEY_EXISTS = -2,
    E_KEY_NOT_FOUND = -3,
    E_POOL_NOT_FOUND = -4,
    E_NOT_SUPPORTED = -5,
    E_ALREADY_EXISTS = -6,
    E_TOO_LARGE = -7,
    E_BAD_PARAM = -8,
  };

  /** 
   * Determine thread safety of the component
   * 
   * 
   * @return THREAD_MODEL_XXXX
   */
  virtual int thread_safety() const = 0;

  /** 
   * Create an object pool
   * 
   * @param path Path of the persistent memory (e.g., /mnt/pmem0/ )
   * @param name Name of object pool
   * @param size Size of object in bytes
   * @param flags Creation flags
   * 
   * @return Pool handle
   */
  virtual pool_t create_pool(const std::string path,
                             const std::string name,
                             const size_t size,
                             unsigned int flags = 0) = 0;

  /** 
   * Open an existing pool
   * 
   * @param path Path of persistent memory (e.g., /mnt/pmem0/ )
   * @param name Name of object pool
   * @param flags Open flags e.g., FLAGS_READ_ONLY
   * 
   * @return Pool handle
   */
  virtual pool_t open_pool(const std::string path,
                           const std::string name,
                           unsigned int flags = 0) = 0;

  /** 
   * Close pool handle
   * 
   * @param pool Pool handle
   */
  virtual void close_pool(const pool_t pool) = 0;

  /** 
   * Delete an existing pool
   * 
   * @param pool Pool handle
   */
  virtual void delete_pool(const pool_t pool)  = 0;

  /** 
   * Write an object value
   * 
   * @param pool Pool handle
   * @param key Object key
   * @param value Value data
   * @param value_len Size of value in bytes
   * 
   * @return S_OK or error code
   */
  virtual int put(const pool_t pool,
                  const std::string key,
                  const void * value,
                  const size_t value_len) = 0;

  /** 
   * Read an object value
   * 
   * @param pool Pool handle
   * @param key Object key
   * @param out_value Value data
   * @param out_value_len Size of value in bytes
   * 
   * @return S_OK or error code
   */
  virtual int get(const pool_t pool,
                  const std::string key,
                  void*& out_value, /* release with free() */
                  size_t& out_value_len) = 0;

  /** 
   * Allocate an object but do not populate data
   * 
   * @param pool Pool handle
   * @param key Key
   * @param nbytes Size to allocate in bytes
   * @param out_key_hash [out] Hash of key
   * 
   * @return S_OK or error code
   */
  virtual int allocate(const pool_t pool,
                       const std::string key,
                       const size_t nbytes,
                       uint64_t& out_key_hash) { return E_NOT_SUPPORTED; }

  /** 
   * Take a lock on an object
   * 
   * @param pool Pool handle
   * @param key_hash Hash of key
   * @param out_value [out] Pointer to data
   * @param out_value_len [out] Size of data in bytes
   * 
   * @return S_OK or error code
   */
  virtual int lock(const pool_t pool,
                   uint64_t key_hash,
                   int type,
                   void*& out_value,
                   size_t& out_value_len) { return E_NOT_SUPPORTED; }

  /** 
   * Unlock an object
   * 
   * @param pool Pool handle
   * @param key_hash Hash of key
   * 
   * @return S_OK or error code
   */
  virtual int unlock(const pool_t pool,
                     uint64_t key_hash) { return E_NOT_SUPPORTED; }
                         
  /** 
   * Apply a functor to an object as a transaction
   * 
   * @param pool Pool handle
   * @param key_hash Hash key (returned from allocate)
   * @param functor Functor to apply to object
   * @param offset Offset within object in bytes
   * @param size Size of window to apply functor on (this changes transaction size)
   * 
   * @return S_OK or error code
   */
  virtual int apply(const pool_t pool,
                    uint64_t key_hash,
                    std::function<void(void*,const size_t)> functor,
                    size_t offset = 0,
                    size_t size = 0) { return E_NOT_SUPPORTED; }
  
  /** 
   * Apply a functor to an object as a transaction
   * 
   * @param pool Pool handle
   * @param key Key
   * @param functor Functor to apply to object
   * @param offset Offset within object in bytes
   * @param size Size of window to apply functor on (this changes transaction size)
   * 
   * @return S_OK or error code
   */
  virtual int apply(const pool_t pool,
                    const std::string key,
                    std::function<void(void*,const size_t)> functor,
                    size_t offset = 0,
                    size_t size = 0) { return E_NOT_SUPPORTED; }

  /** 
   * Apply a functor to an already locked object
   * 
   * @param pool Pool handle
   * @param key Key
   * @param functor Functor to apply to object
   * @param offset Offset within object in bytes
   * @param size Size of window to apply functor on (this changes transaction size)
   * 
   * @return S_OK or error code
   */
  virtual int locked_apply(const pool_t pool,
                           const std::string key,
                           std::function<void(void*,const size_t)> functor,
                           size_t offset = 0,
                           size_t size = 0) { return E_NOT_SUPPORTED; }

  /** 
   * Apply a functor to an already locked object
   * 
   * @param pool Pool handle
   * @param key_hash Hash of key
   * @param functor Functor to apply to object
   * @param offset Offset within object in bytes
   * @param size Size of window to apply functor on (this changes transaction size)
   * 
   * @return S_OK or error code
   */
  virtual int locked_apply(const pool_t pool,
                           uint64_t key_hash,
                           std::function<void(void*,const size_t)> functor,
                           size_t offset = 0,
                           size_t size = 0) { return E_NOT_SUPPORTED; }

  /** 
   * Erase an object
   * 
   * @param pool Pool handle
   * @param key Key
   * 
   * @return S_OK or error code
   */
  virtual int erase(const pool_t pool,
                    const std::string key)= 0;

  /** 
   * Erase an object
   * 
   * @param pool Pool handle
   * @param key_hash Hash of key
   * 
   * @return S_OK or error code
   */  
  virtual int erase(const pool_t pool,
                    uint64_t key_hash) { return E_NOT_SUPPORTED; }


  /** 
   * Return number of objects in the pool
   * 
   * @param pool Pool handle
   * 
   * @return Number of objects
   */
  virtual size_t count(const pool_t pool) = 0;


  /** 
   * Apply functor to all objects in the pool
   * 
   * @param pool Pool handle
   * @param function Functor
   * 
   * @return S_OK or error code
   */
  virtual int map(const pool_t pool,
                  std::function<int(uint64_t key,
                                    const void * value,
                                    const size_t value_len)> function) { return E_NOT_SUPPORTED; }

  /** 
   * Debug routine
   * 
   * @param pool Pool handle
   * @param cmd Debug command
   * @param arg Parameter for debug operation
   */
  virtual void debug(const pool_t pool, unsigned cmd, uint64_t arg) = 0;
};


class IKVStore_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xface829f,0x0405,0x4c19,0x9898,0xa3,0xae,0x21,0x5a,0x3e,0xe8);

  virtual IKVStore * create(const std::string owner,
                            const std::string name){
    throw API_exception("not implemented.");
  }

  virtual IKVStore * create(const std::string owner,
                            const std::string name,  
                             Component::IBlock_device *blk_dev,
                             Component::IBlock_allocator *blk_alloc){
    throw API_exception("not implemented.");
  }
  

};


}


#endif 
