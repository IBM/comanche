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

#ifndef __DAWN_CLIENT_COMPONENT_H__
#define __DAWN_CLIENT_COMPONENT_H__

#include <api/components.h>
#include <api/fabric_itf.h>
#include <api/kvstore_itf.h>
#include <api/kvindex_itf.h>
#include <api/dawn_itf.h>

#include "connection.h"
#include "dawn_client_config.h"

class Dawn_client : public Component::IKVStore,
                    public Component::IDawn
{
  friend class Dawn_client_factory;

 private:
  static constexpr bool option_DEBUG = true;

 protected:
  /**
   * Constructor
   *
   *
   */
  Dawn_client(unsigned           debug_level,
              const std::string& owner,
              const std::string& addr_port_str,
              const std::string& device);

 public:
  /**
   * Destructor
   *
   */
  virtual ~Dawn_client();

  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1);

  // clang-format off
  DECLARE_COMPONENT_UUID(0x2f666078, 0xcb8a, 0x4724, 0xa454, 0xd1, 0xd8, 0x8d, 0xe2, 0xdb, 0x87);
  // clang-format on

  void* query_interface(Component::uuid_t& itf_uuid) override
  {
    if (itf_uuid == Component::IKVStore::iid()) {
      return (void*) static_cast<Component::IKVStore*>(this);
    }
    else if (itf_uuid == Component::IDawn::iid()) {
      return (void*) static_cast<Component::IDawn*>(this);
    }
    else {
      return NULL;  // we don't support this interface
    }
  }

  void unload() override { delete this; }

 public:
  /* IKVStore (as remote proxy) */
  virtual int thread_safety() const override;

  virtual pool_t create_pool(const std::string& path,
                             const std::string& name,
                             const size_t       size,
                             unsigned int       flags    = 0,
                             uint64_t expected_obj_count = 0) override;

  virtual pool_t open_pool(const std::string& path,
                           const std::string& name,
                           unsigned int       flags = 0) override;

  virtual status_t close_pool(const pool_t pool) override;

  virtual status_t delete_pool(const pool_t pool) override;

  virtual status_t put(const pool_t       pool,
                       const std::string& key,
                       const void*        value,
                       const size_t       value_len) override;

  virtual status_t put_direct(const pool_t       pool,
                              const std::string& key,
                              const void*        value,
                              const size_t       value_len,
                              memory_handle_t    handle) override;

  virtual status_t get(const pool_t       pool,
                       const std::string& key,
                       void*&             out_value, /* release with free() */
                       size_t&            out_value_len) override;

  virtual status_t get_direct(const pool_t       pool,
                              const std::string& key,
                              void*              out_value,
                              size_t&            out_value_len,
                              memory_handle_t    handle = HANDLE_NONE) override;

  virtual status_t erase(const pool_t pool, const std::string& key) override;

  virtual size_t count(const pool_t pool) override;

  virtual void debug(const pool_t pool, unsigned cmd, uint64_t arg) override;

  virtual Component::IKVStore::memory_handle_t register_direct_memory(void*  vaddr,
                                                                      size_t len) override;

  virtual status_t unregister_direct_memory(memory_handle_t handle) override;

  virtual status_t free_memory(void * p) override;

  /* IDawn specific methods */
  virtual pool_t create_pool(const std::string& pool_name,
                             const size_t size,
                             unsigned int flags = 0,
                             uint64_t expected_obj_count = 0) override;

  virtual pool_t open_pool(const std::string& pool_name,
                           unsigned int flags = 0) override;

  virtual std::string find(const std::string& key_expression,
                           Component::IKVIndex::offset_t begin_position,
                           Component::IKVIndex::find_t find_type,
                           Component::IKVIndex::offset_t& out_end_position) override;

  
 private:

  Component::IFabric_factory* _factory;
  Component::IFabric*         _fabric;
  Component::IFabric_client*  _transport;

  Dawn::Client::Connection_handler* _connection;

 private:
  void open_transport(const std::string& device,
                      const std::string& ip_addr,
                      const int          port,
                      const std::string& provider);

  void close_transport();
};


class Dawn_client_factory : public Component::IKVStore_factory {
 public:
  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1);

  // clang-format off
  DECLARE_COMPONENT_UUID(0xfac66078, 0xcb8a, 0x4724, 0xa454, 0xd1, 0xd8, 0x8d, 0xe2, 0xdb, 0x87);
  // clang-format on

  void* query_interface(Component::uuid_t& itf_uuid) override
  {
    if (itf_uuid == Component::IKVStore_factory::iid()) {
      return (void*) static_cast<Component::IKVStore_factory*>(this);
    }
    else
      return NULL;  // we don't support this interface
  }

  void unload() override { delete this; }

  virtual Component::IKVStore* create(unsigned           debug_level,
                                      const std::string& owner,
                                      const std::string& addr,
                                      const std::string& param2) override
  {
    Component::IKVStore* obj = static_cast<Component::IKVStore*>(new Dawn_client(debug_level, owner, addr, param2));
    obj->add_ref();
    return obj;
  }
};

#endif
