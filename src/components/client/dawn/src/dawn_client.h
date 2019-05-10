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

#ifndef __DAWN_CLIENT_COMPONENT_H__
#define __DAWN_CLIENT_COMPONENT_H__

#include <api/components.h>
#include <api/fabric_itf.h>
#include <api/kvstore_itf.h>
#include <api/kvindex_itf.h>
#include <api/dawn_itf.h>

#include "connection.h"
#include "dawn_client_config.h"

class Dawn_client : public virtual Component::IKVStore,
                    public virtual Component::IDawn
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
  /** 
   * Constructor
   * 
   * @param debug_level Debug level (e.g., 0-3)
   * @param owner Owner information (not used)
   * @param addr_port_str Address and port info (e.g. 10.0.0.22:11911)
   * @param device NIC device (e.g., mlnx5_0)
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

  virtual int get_capability(Capability cap) const override;

  virtual pool_t create_pool(const std::string& name,
                             const size_t       size,
                             uint32_t           flags    = 0,
                             uint64_t expected_obj_count = 0) override;

  virtual pool_t open_pool(const std::string& name,
                           uint32_t           flags = 0) override;

  virtual status_t close_pool(const pool_t pool) override;

  virtual status_t delete_pool(const std::string& name) override;

  virtual status_t configure_pool(const Component::IKVStore::pool_t pool,
                                  const std::string& json) override;

  virtual status_t put(const pool_t       pool,
                       const std::string& key,
                       const void*        value,
                       const size_t       value_len,
                       uint32_t       flags = FLAGS_NONE) override;

  virtual status_t put_direct(const pool_t       pool,
                              const std::string& key,
                              const void*        value,
                              const size_t       value_len,
                              memory_handle_t    handle = HANDLE_NONE,
                              uint32_t           flags = FLAGS_NONE) override;

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

  virtual status_t get_attribute(const IKVStore::pool_t pool,
                                 const IKVStore::Attribute attr,
                                 std::vector<uint64_t>& out_attr,
                                 const std::string* key) override;

  virtual void debug(const pool_t pool, unsigned cmd, uint64_t arg) override;

  virtual Component::IKVStore::memory_handle_t register_direct_memory(void*  vaddr,
                                                                      size_t len) override;

  virtual status_t unregister_direct_memory(memory_handle_t handle) override;

  virtual status_t free_memory(void * p) override;

  /* IDawn specific methods */
  virtual status_t find(const IKVStore::pool_t pool,
                        const std::string& key_expression,
                        const offset_t offset,
                        offset_t& out_matched_offset,
                        std::string& out_matched_key) override;
  
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


class Dawn_client_factory : public Component::IDawn_factory
{
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
    if (itf_uuid == Component::IDawn_factory::iid()) {
      return (void*) static_cast<Component::IDawn_factory*>(this);
    }
    else if (itf_uuid == Component::IKVStore_factory::iid()) {
      return (void*) static_cast<Component::IKVStore_factory*>(this);
    }
    else return NULL;  // we don't support this interface
  }

  void unload() override { delete this; }

  Component::IDawn * dawn_create(unsigned           debug_level,
                                 const std::string& owner,
                                 const std::string& addr,
                                 const std::string& device) override
  {
    Component::IDawn* obj =
      static_cast<Component::IDawn*>(new Dawn_client(debug_level, owner, addr, device));
    obj->add_ref();
    return obj;
  }
  
  Component::IKVStore * create(unsigned           debug_level,
                               const std::string& owner,
                               const std::string& addr,
                               const std::string& device) override
  {
    Component::IKVStore* obj =
      static_cast<Component::IKVStore*>(new Dawn_client(debug_level, owner, addr, device));
    obj->add_ref();
    return obj;
  }

};

#endif
