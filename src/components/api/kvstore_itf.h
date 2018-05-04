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

#include <api/components.h>

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
    FLAGS_READ_ONLY,
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
    
  virtual int thread_safety() const = 0;
    
  virtual pool_t create_pool(const std::string path,
                             const std::string name,
                             const size_t size,
                             unsigned int flags = 0) = 0;

  virtual pool_t open_pool(const std::string path,
                           const std::string name,
                           unsigned int flags = 0) = 0;

  virtual void close_pool(const pool_t pid) = 0;

  virtual int put(const pool_t pool,
                  const std::string key,
                  const void * value,
                  const size_t value_len) = 0;

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
   * @return S_OK on success
   */
  virtual int allocate(const pool_t pool,
                       const std::string key,
                       const size_t nbytes,
                       uint64_t& out_key_hash) { return E_NOT_SUPPORTED; }

  /** 
   * Apply a functor to an object as a transaction
   * 
   * @param pool Pool handle
   * @param key_hash Hash key (returned from allocate)
   * @param functor Functor to apply to object
   * @param offset
   * @param size
   * 
   * @return S_OK on success
   */
  virtual int apply(const pool_t pool,
                    uint64_t key_hash,
                    std::function<void(void*,const size_t)> functor,
                    size_t offset = 0,
                    size_t size = 0) { return E_NOT_SUPPORTED; }

  virtual int apply(const pool_t pool,
                    const std::string key,
                    std::function<void(void*,const size_t)> functor,
                    size_t offset = 0,
                    size_t size = 0) { return E_NOT_SUPPORTED; }

                               
  virtual int remove(const pool_t pool,
                     const std::string key)= 0;
  
  virtual int remove(const pool_t pool,
                     uint64_t key_hash) { return E_NOT_SUPPORTED; }


  virtual size_t count(const pool_t pool) = 0;

  virtual int map(const pool_t pool,
                  std::function<int(uint64_t key,
                                    const void * value,
                                    const size_t value_len)> function) { return E_NOT_SUPPORTED; }
  
  virtual void debug(const pool_t pool, unsigned cmd, uint64_t arg) = 0;
};


class IKVStore_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xface829f,0x0405,0x4c19,0x9898,0xa3,0xae,0x21,0x5a,0x3e,0xe8);

  virtual IKVStore * create(const std::string owner,
                            const std::string name) = 0;
  

};


}


#endif 
