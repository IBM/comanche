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

#ifndef __KVSTORE_COMPONENT_H__
#define __KVSTORE_COMPONENT_H__

#include <unordered_map>
#include <pthread.h>
#include <common/rwlock.h>
#include <tbb/concurrent_hash_map.h>
#include <api/kvstore_itf.h>

class PM_store : public Component::IKVStore
{
private:
  bool option_DEBUG;

public:
  /** 
   * Constructor
   * 
   * @param block_device Block device interface
   * 
   */
  PM_store(unsigned int debug_level, const std::string& owner, const std::string& name);

  /** 
   * Destructor
   * 
   */
  virtual ~PM_store();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x59564581,0x9e1b,0x4811,0xbdb2,0x19,0x57,0xa0,0xa6,0x84,0x57);


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
  virtual status_t thread_safety() const { return THREAD_MODEL_SINGLE_PER_POOL; }

  virtual pool_t create_pool(const std::string& path,
                             const std::string& name,
                             const size_t size,
                             unsigned int flags,
                             uint64_t expected_obj_count = 0
                             ) override;

  virtual pool_t open_pool(const std::string& path,
                           const std::string& name,
                           unsigned int flags) override;

  virtual void delete_pool(const pool_t pid) override;

  virtual void close_pool(const pool_t pid) override;

  virtual status_t get_pool_regions(const pool_t pool, std::vector<::iovec>& out_regions) override;
  
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
                              Component::IKVStore::memory_handle_t handle = nullptr) override;

  virtual status_t put_direct(const pool_t pool,
                              const std::string& key,
                              const void * value,
                              const size_t value_len,
                              memory_handle_t handle = HANDLE_NONE) override;

  virtual Component::IKVStore::memory_handle_t register_direct_memory(void * vaddr, size_t len) override;

  virtual Component::IKVStore::key_t lock(const pool_t pool,
                                          const std::string& key,
                                          lock_type_t type,
                                          void*& out_value,
                                          size_t& out_value_len) override;

  virtual status_t unlock(const pool_t pool,
                          Component::IKVStore::key_t key_handle) override;

  virtual status_t apply(const pool_t pool,
                         const std::string& key,
                         std::function<void(void*,const size_t)> functor,
                         size_t object_size,
                         bool use_lock = true) override;

  virtual status_t erase(const pool_t pool,
                         const std::string& key) override;

  virtual size_t count(const pool_t pool) override;

  // virtual status_t map(const pool_t pool,
  //                      std::function<int(const std::string& key,
  //                                        const void * value,
  //                                        const size_t value_len)> function) P

  virtual void debug(const pool_t pool, unsigned cmd, uint64_t arg) override;

private:

  virtual status_t __apply(const pool_t pool,
                           uint64_t key_hash,
                           std::function<void(void*,const size_t)> functor,
                           size_t object_size,
                           bool take_lock);

private:

  struct volatile_state_t
  {
    Common::RWLock _lock;
  };

  class State_map
  {
    using pool_state_map_t = std::unordered_map<const void*, volatile_state_t>;
    /* we use a concurrent/thread-safe map so we can support multiple
       threads on different pools
       TODO: cleaning up out pool entries? */
    using state_map_t = tbb::concurrent_hash_map<const pool_t, pool_state_map_t>;

  public:
    bool state_get_read_lock(const pool_t pool, const void * ptr);
    bool state_get_write_lock(const pool_t pool, const void * ptr);
    void state_unlock(const pool_t pool, const void * ptr);
    void state_remove(const pool_t pool, const void * ptr);
  private:
    state_map_t _state_map;
  };

  State_map _sm;

};


class PM_store_factory : public Component::IKVStore_factory
{
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac64581,0x9e1b,0x4811,0xbdb2,0x19,0x57,0xa0,0xa6,0x84,0x57);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IKVStore_factory::iid()) {
      return (void *) static_cast<Component::IKVStore_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  virtual Component::IKVStore * create(unsigned int debug_level,
				       const std::string& owner,
                                       const std::string& name,
				       const std::string& param2) override
  {
    Component::IKVStore * obj = static_cast<Component::IKVStore*>(new PM_store(debug_level, owner, name));
    obj->add_ref();
    return obj;
  }

};



#endif
