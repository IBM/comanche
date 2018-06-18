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

#ifndef __FILESTORE_COMPONENT_H__
#define __FILESTORE_COMPONENT_H__

#include <api/kvstore_itf.h>

class FileStore : public Component::IKVStore /* generic Key-Value store interface */
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
  FileStore(const std::string owner, const std::string name);

  /** 
   * Destructor
   * 
   */
  virtual ~FileStore();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x8a120985,0xe253,0x404d,0x94d7,0x77,0x92,0x75,0x22,0xa9,0x20);

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
  virtual int thread_safety() const { return THREAD_MODEL_MULTI_PER_POOL; }
  
  virtual pool_t create_pool(const std::string path,
                             const std::string name,
                             const size_t size,
                             unsigned int flags = 0,
                             uint64_t expected_obj_count = 0) override;
  
  virtual pool_t open_pool(const std::string path,
                           const std::string name,
                           unsigned int flags = 0) override;
  
  virtual void close_pool(const pool_t pid) override;

  virtual void delete_pool(const pool_t pid) override;

  virtual int put(const pool_t pool,
                  const std::string key,
                  const void * value,
                  const size_t value_len) override;

  virtual int get(const pool_t pool,
                  const std::string key,
                  void*& out_value,
                  size_t& out_value_len) override;
  
  virtual int get_direct(const pool_t pool,
                         const std::string key,
                         void* out_value,
                         size_t out_value_len) override;
  
  virtual int erase(const pool_t pool,
                    const std::string key) override;

  virtual size_t count(const pool_t pool) override;

  virtual void debug(const pool_t pool, unsigned cmd, uint64_t arg) override;
  
private:
  
};


class FileStore_factory : public Component::IKVStore_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac20985,0xe253,0x404d,0x94d7,0x77,0x92,0x75,0x22,0xa9,0x20);
  
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
    Component::IKVStore * obj = static_cast<Component::IKVStore*>(new FileStore(owner, name));    
    obj->add_ref();
    return obj;
  }

};



#endif
