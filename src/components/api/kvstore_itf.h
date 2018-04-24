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
 * Kve-value interface
 */
class IKVStore : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0x62f4829f,0x0405,0x4c19,0x9898,0xa3,0xae,0x21,0x5a,0x3e,0xe8);

public:
  using iterator_t = void*;
  using pool_t     = void*;

  enum {
    THREAD_MODEL_UNSAFE,
    THREAD_MODEL_SINGLE_PER_POOL,
    THREAD_MODEL_MULTI_PER_POOL,
  };
    
  virtual int thread_safety() const = 0;
    
  virtual pool_t create_pool(const std::string path,
                            const std::string name,
                            const size_t size) = 0;

  virtual pool_t open_pool(const std::string path,
                          const std::string name) = 0;

  virtual void close_pool(const pool_t pid) = 0;

  virtual void put(const pool_t pool,
                   const std::string key,
                   const void * value,
                   const size_t value_len) = 0;

  virtual void get(const pool_t pool,
                   const std::string key,
                   void*& out_value, /* release with free() */
                   size_t& out_value_len) = 0;

  virtual void get_reference(const pool_t pool,
                             const std::string key,
                             const void*& out_value,
                             size_t& out_value_len) = 0;

  virtual void release_reference(const pool_t pool,
                                 const void * ptr) = 0;
                             
  virtual void remove(const pool_t pool,
                      const std::string key)= 0;

  virtual size_t count(const pool_t pool) = 0;

  virtual void apply(const pool_t pool,
                     std::function<int(uint64_t key,
                                       const void * value,
                                       const size_t value_len)> function) = 0;

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
