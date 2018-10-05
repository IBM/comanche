/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 *
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
  virtual ~Map_store();

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
  
  virtual pool_t create_pool(const std::string& path,
                             const std::string& name,
                             const size_t size,
                             unsigned int flags = 0,
                             uint64_t expected_obj_count = 0) override;
  
  virtual pool_t open_pool(const std::string& path,
                           const std::string& name,
                           unsigned int flags = 0) override;
  
  virtual void close_pool(const pool_t pid) override;

  virtual void delete_pool(const pool_t pid) override;

  virtual status_t put(const pool_t pool,
                       const std::string& key,
                       const void * value,
                       const size_t value_len) override;

  virtual status_t get(const pool_t pool,
                       const std::string& key,
                       void*& out_value,
                       size_t& out_value_len) override;
  
  virtual status_t get_direct(const pool_t pool,
                              const std::string& key,
                              void* out_value,
                              size_t& out_value_len,
                              size_t offset,
                              Component::IKVStore::memory_handle_t handle) override;
 
  virtual status_t put_direct(const pool_t pool,
                              const std::string& key,
                              const void * value,
                              const size_t value_len,
                              size_t offset = 0,
                              IKVStore::memory_handle_t handle = HANDLE_NONE) override;
  
  virtual status_t lock(const pool_t pool,
                        uint64_t key_hash,
                        lock_type_t type,
                        void*& out_value,
                        size_t& out_value_len) override;

  virtual status_t unlock(const pool_t pool,
                          uint64_t key_hash) override;

  virtual status_t erase(const pool_t pool,
                         const std::string& key) override;
  
  virtual size_t count(const pool_t pool) override;

  virtual void free_memory(void * p) override;
  
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
