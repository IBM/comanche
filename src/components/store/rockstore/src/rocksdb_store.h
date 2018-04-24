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

#include <api/kvstore_itf.h>

class RockStore : public Component::IKVStore /* generic Key-Value store interface */
{  
private:
  static constexpr bool option_DEBUG = true;

public:
  /** 
   * Constructor
   * 
   * @param block_device Block device interface
   * 
   */
  RockStore(const std::string owner, const std::string name);

  /** 
   * Destructor
   * 
   */
  virtual ~RockStore();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x0a1781e5,0x2db9,0x4876,0xb492,0xe2,0x5b,0xfe,0x17,0x3a,0xac);

  
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

  /* IRockStore */
  virtual int thread_safety() const { return THREAD_MODEL_MULTI_PER_POOL; }
  
  virtual pool_t create_pool(const std::string path,
                             const std::string name,
                             const size_t size) override;
  
  virtual pool_t open_pool(const std::string path,
                           const std::string name) override;
  
  virtual void close_pool(const pool_t pid) override;

  virtual void put(const pool_t pool,
                   const std::string key,
                   const void * value,
                   const size_t value_len) override;

  virtual void get(const pool_t pool,
                   const std::string key,
                   void*& out_value,
                   size_t& out_value_len) override;
  
  virtual void get_reference(const pool_t pool,
                             const std::string key,
                             const void*& out_value,
                             size_t& out_value_len) override;
  
  virtual void release_reference(const pool_t pool,
                                 const void * ptr) override;


  virtual void remove(const pool_t pool,
                      const std::string key) override;

  virtual size_t count(const pool_t pool) override;

  virtual void apply(const pool_t pool,
                     std::function<int(uint64_t key,
                                       const void * value,
                                       const size_t value_len)> function) override;

  virtual void debug(const pool_t pool, unsigned cmd, uint64_t arg) override;
  
private:
  
};


class RockStore_factory : public Component::IKVStore_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac781e5,0x2db9,0x4876,0xb492,0xe2,0x5b,0xfe,0x17,0x3a,0xac);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IKVStore_factory::iid()) {
      return (void *) static_cast<Component::IKVStore_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  virtual Component::IKVStore * create(const std::string owner,
                                       const std::string name) override
  {    
    Component::IKVStore * obj = static_cast<Component::IKVStore*>(new RockStore(owner, name));    
    obj->add_ref();
    return obj;
  }

};



#endif
