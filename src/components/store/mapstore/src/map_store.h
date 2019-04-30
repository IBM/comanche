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


/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __MAP_STORE_COMPONENT_H__
#define __MAP_STORE_COMPONENT_H__

#include <api/kvstore_itf.h>

class Map_store : public Component::IKVStore /* generic Key-Value store interface */
{  
private:
  static constexpr bool option_DEBUG = false;

public:
  /** 
   * Constructor
   * 
   * @param block_device Block device interface
   * 
   */
  Map_store(const std::string& owner, const std::string& name);

  /** 
   * Destructor
   * 
   */
  ~Map_store();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x8a120985,0x1253,0x404d,0x94d7,0x77,0x92,0x75,0x21,0xa1,0x21);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IKVStore::iid()) {
      return (void *) static_cast<Component::IKVStore*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

public:

  /* IKVStore */
  virtual int thread_safety() const { return THREAD_MODEL_RWLOCK_PER_POOL; }

  virtual int get_capability(Capability cap) const;
  
  virtual pool_t create_pool(const std::string& name,
                             const size_t size,
                             unsigned int flags = 0,
                             uint64_t expected_obj_count = 0) override;
  
  virtual pool_t open_pool(const std::string& name,
                           unsigned int flags = 0) override;
  
  virtual status_t close_pool(const pool_t pid) override;

  virtual status_t delete_pool(const std::string& name) override;

  virtual status_t put(const pool_t pool,
                       const std::string& key,
                       const void * value,
                       const size_t value_len,
                       unsigned int flags = FLAGS_NONE) override;

  virtual status_t get(const pool_t pool,
                       const std::string& key,
                       void*& out_value,
                       size_t& out_value_len) override;
  
  virtual status_t get_direct(const pool_t pool,
                              const std::string& key,
                              void* out_value,
                              size_t& out_value_len,
                              Component::IKVStore::memory_handle_t handle) override;
 
  virtual status_t put_direct(const pool_t pool,
                              const std::string& key,
                              const void * value,
                              const size_t value_len,
                              IKVStore::memory_handle_t handle = HANDLE_NONE,
                              unsigned int flags = FLAGS_NONE) override;

  virtual status_t get_attribute(const pool_t pool,
                                 const Attribute attr,
                                 std::vector<uint64_t>& out_attr,
                                 const std::string* key = nullptr) override;

  virtual key_t lock(const pool_t pool,
                     const std::string& key,
                     lock_type_t type,
                     void*& out_value,
                     size_t& out_value_len) override;

  virtual status_t unlock(const pool_t pool,
                          key_t key) override;

  virtual status_t erase(const pool_t pool,
                         const std::string& key) override;
  
  virtual size_t count(const pool_t pool) override;

  virtual status_t free_memory(void * p) override;

  virtual status_t map(const pool_t pool,
                       std::function<int(const std::string& key,
                                         const void * value,
                                         const size_t value_len)> function) override;

  virtual void debug(const pool_t pool, unsigned cmd, uint64_t arg) override;

private:
  
};


class Map_store_factory : public Component::IKVStore_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac20985,0x1253,0x404d,0x94d7,0x77,0x92,0x75,0x21,0xa1,0x21);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IKVStore_factory::iid()) {
      return (void *) static_cast<Component::IKVStore_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  virtual Component::IKVStore * create(const std::string& owner,
                                       const std::string& name) override
  {    
    Component::IKVStore * obj = static_cast<Component::IKVStore*>(new Map_store(owner, name));
    assert(obj);
    obj->add_ref();
    return obj;
  }

};



#endif
