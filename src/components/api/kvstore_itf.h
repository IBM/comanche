/*
   Copyright [2017,2018] [IBM Corporation]

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
    FLAGS_SET_SIZE = 2,
  };

  enum {
    STORE_LOCK_READ=1,
    STORE_LOCK_WRITE=2,
  };

  enum {
    S_OK = 0,
    S_MORE = 1,
    E_FAIL = -1,
    E_KEY_EXISTS = -2,
    E_KEY_NOT_FOUND = -3,
    E_POOL_NOT_FOUND = -4,
    E_NOT_SUPPORTED = -5,
    E_ALREADY_EXISTS = -6,
    E_TOO_LARGE = -7,
    E_BAD_PARAM = -8,
    E_BAD_ALIGNMENT = -9,
    E_INSUFFICIENT_BUFFER = -10,
    E_BAD_OFFSET = -11,    
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
   * @param
   * 
   * @return Pool handle
   */
  virtual pool_t create_pool(const std::string path,
                             const std::string name,
                             const size_t size,
                             unsigned int flags = 0,
                             uint64_t expected_obj_count = 0) = 0;

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
  virtual status_t put(const pool_t pool,
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
  virtual status_t get(const pool_t pool,
                       const std::string key,
                       void*& out_value, /* release with free() */
                       size_t& out_value_len) = 0;


  /** 
   * Read an object value directly into client-provided memory.  To perform partial gets you can 
   * use the offset parameter and limit the size of the buffer (out_value_len).  Loop on return of
   * S_MORE, and increment offset, to read fragments.  This is useful for very large objects that 
   * for instance you want to start sending over the network while the device is pulling the data in.
   * 
   * @param pool Pool handle
   * @param key Object key
   * @param out_value Client provided buffer for value
   * @param out_value_len [in] size of value memory in bytes [out] size of value
   * @param offset Offset from beginning of value in bytes.
   * 
   * @return S_OK, S_MORE if only a portion of value is read, E_BAD_ALIGNMENT on invalid alignment, or other error code
   */
  virtual status_t get_direct(const pool_t pool,
                              const std::string key,
                              void* out_value,
                              size_t& out_value_len,
                              size_t offset = 0) { return E_NOT_SUPPORTED; }


  /** 
   * Register memory for zero copy DMA
   * 
   * @param vaddr Appropriately aligned memory buffer
   * @param len Length of memory buffer in bytes
   * 
   * @return S_OK on success
   */
  virtual status_t register_direct_memory(void * vaddr, size_t len) { return E_NOT_SUPPORTED; }
  

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
  virtual status_t allocate(const pool_t pool,
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
  virtual status_t lock(const pool_t pool,
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
  virtual status_t unlock(const pool_t pool,
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
  virtual status_t apply(const pool_t pool,
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
  virtual status_t apply(const pool_t pool,
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
  virtual status_t locked_apply(const pool_t pool,
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
  virtual status_t locked_apply(const pool_t pool,
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
  virtual status_t erase(const pool_t pool,
                         const std::string key)= 0;
  
  /** 
   * Erase an object
   * 
   * @param pool Pool handle
   * @param key_hash Hash of key
   * 
   * @return S_OK or error code
   */  
  virtual status_t erase(const pool_t pool,
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
  virtual status_t map(const pool_t pool,
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
    throw(API_exception("Not Implemented"));
  };

  virtual IKVStore * create(const std::string owner,
                            const std::string name,
                            std::string pci){
    throw(API_exception("Not Implemented"));
  }

};


}


#endif 
