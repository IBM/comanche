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


#ifndef __API_DAWN_ITF__
#define __API_DAWN_ITF__


#include <api/components.h>
#include <api/kvstore_itf.h>
#include <api/kvindex_itf.h>

namespace Component
{

/** 
 * Dawn client interface (this will include both KV and AS capabilities)
 */
class IDawn : public Component::IBase
{
public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0x33af1b99,0xbc51,0x49ff,0xa27b,0xd4,0xe8,0x19,0x03,0xbb,0x02);
  // clang-format on
  
public:
  

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
   * @param ppol_name Unique pool name
   * @param size Size of pool in bytes
   * @param flags Creation flags
   * @param expected_obj_count Expected maximum object count (optimization)
   * 
   * @return Pool handle
   */
  virtual IKVStore::pool_t create_pool(const std::string& pool_name,
                                       const size_t size,
                                       unsigned int flags = 0,
                                       uint64_t expected_obj_count = 0) = 0;
  
  /** 
   * Open an existing pool
   * 
   * @param pool_name Name of object pool
   * @param flags Open flags e.g., FLAGS_READ_ONLY
   * 
   * @return Pool handle
   */
  virtual IKVStore::pool_t open_pool(const std::string& pool_name,
                                     unsigned int flags = 0) = 0;

  /** 
   * Close pool handle
   * 
   * @param pool Pool handle
   */
  virtual status_t close_pool(const IKVStore::pool_t pool) = 0;

  /** 
   * Close and delete an existing pool
   * 
   * @param pool Pool name
   */
  virtual status_t delete_pool(const std::string& pool_name) = 0;

  /** 
   * Write or overwrite an object value. If there already exists a
   * object with matching key, then it should be replaced
   * (i.e. reallocated) or overwritten. 
   * 
   * @param pool Pool handle
   * @param key Object key
   * @param value Value data
   * @param value_len Size of value in bytes
   * @param flags Additional flags
   * 
   * @return S_OK or error code
   */
  virtual status_t put(const IKVStore::pool_t pool,
                       const std::string& key,
                       const void * value,
                       const size_t value_len,
                       unsigned int flags = IKVStore::FLAGS_NONE) = 0;

  /** 
   * Zero-copy put operation.  If there does not exist an object
   * with matching key, then an error E_KEY_EXISTS should be returned.
   * 
   * @param pool Pool handle
   * @param key Object key
   * @param key_len Key length in bytes
   * @param value Value
   * @param value_len Value length in bytes
   * @param handle Memory registration handle
   * @param flags Additional flags 
   *
   * @return S_OK or error code
   */
  virtual status_t put_direct(const IKVStore::pool_t pool,
                              const std::string& key,
                              const void * value,
                              const size_t value_len,
                              IKVStore::IKVStore::memory_handle_t handle = IKVStore::HANDLE_NONE,
                              unsigned int flags = IKVStore::FLAGS_NONE) = 0;

  /** 
   * Read an object value
   * 
   * @param pool Pool handle
   * @param key Object key
   * @param out_value Value data (if null, component will allocate memory)
   * @param out_value_len Size of value in bytes
   * 
   * @return S_OK or error code
   */
  virtual status_t get(const IKVStore::pool_t pool,
                       const std::string& key,
                       void*& out_value, /* release with free_memory() API */
                       size_t& out_value_len) = 0;


  /**
   * Read an object value directly into client-provided memory.
   *
   * @param pool Pool handle
   * @param key Object key
   * @param out_value Client provided buffer for value
   * @param out_value_len [in] size of value memory in bytes [out] size of value
   * @param handle Memory registration handle
   * 
   * @return S_OK, S_MORE if only a portion of value is read, E_BAD_ALIGNMENT on invalid alignment, or other error code
   */
  virtual status_t get_direct(const IKVStore::pool_t pool,
                              const std::string& key,
                              void* out_value,
                              size_t& out_value_len,
                              IKVStore::memory_handle_t handle = IKVStore::HANDLE_NONE) = 0;
  
  /** 
   * Perform a key search
   * 
   * @param key_expression Key expression to match on
   * @param begin_position Position from which to start from. Counting from 0.
   * @param out_end_position [out] Position of the match
   * 
   * @return Matched key
   */  
  virtual std::string find(const std::string& key_expression,
                           Component::IKVIndex::offset_t begin_position,
                           Component::IKVIndex::find_t find_type,
                           Component::IKVIndex::offset_t& out_end_position) = 0;

  /** 
   * Erase an object
   * 
   * @param pool Pool handle
   * @param key Object key
   * 
   * @return S_OK or error code
   */
  virtual status_t erase(const IKVStore::pool_t pool,
                         const std::string& key)= 0;

  /** 
   * Return number of objects in the pool
   * 
   * @param pool Pool handle
   * 
   * @return Number of objects
   */
  virtual size_t count(const IKVStore::pool_t pool) = 0;

  /** 
   * Register memory for zero copy DMA
   * 
   * @param vaddr Appropriately aligned memory buffer
   * @param len Length of memory buffer in bytes
   * 
   * @return Memory handle or NULL on not supported.
   */
  virtual IKVStore::memory_handle_t register_direct_memory(void * vaddr, size_t len) = 0;

  /** 
   * Durict memory regions should be unregistered before the memory is released on the client side.
   * 
   * @param vaddr Address of region to unregister.
   * 
   * @return S_OK on success
   */
  virtual status_t unregister_direct_memory(IKVStore::memory_handle_t handle) = 0;

  /**
   * Free API allocated memory
   *
   * @param p Pointer to memory allocated through a get call
   * 
   * @return S_OK on success
   */
  virtual status_t free_memory(void * p) = 0;

  /** 
   * Debug routine
   * 
   * @param pool Pool handle
   * @param cmd Debug command
   * @param arg Parameter for debug operation
   */
  virtual void debug(const IKVStore::pool_t pool, unsigned cmd, uint64_t arg) = 0;
};


class IDawn_factory : public IKVStore_factory
{
public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xfacf1b99,0xbc51,0x49ff,0xa27b,0xd4,0xe8,0x19,0x03,0xbb,0x02);
  // clang-format on
  
  virtual IDawn * dawn_create(const std::string& owner,
                            const std::string& param){
    throw API_exception("factory::create(owner,param) not implemented");
  };

  virtual IDawn * dawn_create(const std::string& owner,
                            const std::string& param,
                            const std::string& param2){
    throw API_exception("factory::create(owner,param,param2) not implemented");
  }

  virtual IDawn * dawn_create(unsigned debug_level,
                            const std::string& owner,
                            const std::string& param,
                            const std::string& param2){
    throw API_exception("factory::create(debug_level,owner,param,param2) not implemented");
  }


};


}


#endif
