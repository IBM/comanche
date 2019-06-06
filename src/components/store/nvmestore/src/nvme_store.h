/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 *
 */

/*
 * Authors:
 *
 * Feng Li (fengggli@yahoo.com)
 */

#ifndef NVME_STORE_H_
#define NVME_STORE_H_

#include <libpmemobj.h>

#include <common/rwlock.h>
#include <common/types.h>
#include <pthread.h>

#include <api/kvstore_itf.h>

#include "block_manager.h"
#include "state_map.h"

#undef USE_PMEM

using namespace nvmestore;

class State_map;

#ifdef USE_PMEM
POBJ_LAYOUT_BEGIN(nvme_store);
POBJ_LAYOUT_ROOT(nvme_store, struct store_root_t);
POBJ_LAYOUT_TOID(nvme_store, struct obj_info);
POBJ_LAYOUT_TOID(nvme_store, char);
POBJ_LAYOUT_END(nvme_store);

/*
 * for block allocator
 */
typedef struct obj_info {
  // Block alocation
  int   lba_start;
  int   size;    // value size in bytes
  void* handle;  // handle to free this block

  // key info
  size_t key_len;
  TOID(char) key_data;
} obj_info_t;
#endif

class NVME_store : public Component::IKVStore {
  using block_manager_t = nvmestore::Block_manager;
  using io_buffer_t     = block_manager_t::io_buffer_t;
  static constexpr size_t DEFAULT_IO_MEM_SIZE =
      MB(8);  // initial IO memory size in bytes

 private:
  static constexpr bool option_DEBUG = true;
  std::string           _pm_path;

  State_map       _sm;           // map control TODO: change to session manager
  block_manager_t _blk_manager;  // shared across all nvmestore
  enum {
    PERSIST_PMEM,
    PERSIST_HSTORE,
    PERSIST_FILE
  } _meta_persist_type; /** How to persist meta data*/
  IKVStore* _meta_store = nullptr;

 public:
  /**
   * Constructor
   *
   * @param owner
   * @param name
   * @param pci pci address of the Nvme
   *   The "pci address" is in Bus:Device.Function (BDF) form with Bus and
   * @param config_persist_type Use either "pmem"(pmdk tx), "hstore",
   * "filestore" to manage metadata, default is pmem
   * Device zero-padded to 2
   * digits each, e.g. 86:00.0
   */
  NVME_store(const std::string& owner,
             const std::string& name,
             const std::string& pci,
             const std::string& pm_path,
             const std::string& config_persist_type);

  /**
   * Destructor
   *
   */
  virtual ~NVME_store();

  /**
   * Component/interface management
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x59564581,
                         0x9e1b,
                         0x4811,
                         0xbdb2,
                         0x19,
                         0x57,
                         0xa0,
                         0xa6,
                         0x84,
                         0x57);

  void* query_interface(Component::uuid_t& itf_uuid) override
  {
    if (itf_uuid == Component::IKVStore::iid()) {
      return (void*) static_cast<Component::IKVStore*>(this);
    }
    else
      return NULL;  // we don't support this interface
  }

  void unload() override { delete this; }

 public:
  /* IKVStore */
  virtual int thread_safety() const override
  {
    return THREAD_MODEL_SINGLE_PER_POOL;
  }

  virtual pool_t create_pool(const std::string& name,
                             const size_t       size,
                             unsigned int       flags,
                             uint64_t expected_obj_count = 0) override;

  virtual pool_t open_pool(const std::string& name,
                           unsigned int       flags) override;

  virtual status_t delete_pool(const std::string& name) override;

  virtual status_t close_pool(const pool_t pid) override;

  virtual status_t put(const pool_t       pool,
                       const std::string& key,
                       const void*        value,
                       const size_t       value_len,
                       unsigned int       flags = FLAGS_NONE) override;

  virtual status_t get(const pool_t       pool,
                       const std::string& key,
                       void*&             out_value,
                       size_t&            out_value_len) override;

  virtual status_t get_direct(
      const pool_t                         pool,
      const std::string&                   key,
      void*                                out_value,
      size_t&                              out_value_len,
      Component::IKVStore::memory_handle_t handle) override;

  virtual memory_handle_t allocate_direct_memory(void*& vaddr,
                                                 size_t len) override;

  virtual status_t free_direct_memory(memory_handle_t handle) override;

  virtual IKVStore::memory_handle_t register_direct_memory(void*  vaddr,
                                                           size_t len) override;

  virtual status_t unregister_direct_memory(memory_handle_t handle) override;

  virtual status_t lock(const pool_t       pool,
                        const std::string& key,
                        lock_type_t        type,
                        void*&             out_value,
                        size_t&            out_value_len,
                        key_t&             out_key) override;

  virtual status_t unlock(const pool_t pool, key_t key_handle) override;

  virtual status_t erase(const pool_t pool, const std::string& key) override;

  virtual size_t   count(const pool_t pool) override;
  virtual status_t map(
      const pool_t                               pool,
      std::function<int(const std::string& key,
                        const void*        value,
                        const size_t       value_len)> function) override;
  virtual status_t map_keys(
      const pool_t                               pool,
      std::function<int(const std::string& key)> function);
  virtual void debug(const pool_t pool, unsigned cmd, uint64_t arg) override;
};

class NVME_store_factory : public Component::IKVStore_factory {
 public:
  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac64581,
                         0x1993,
                         0x4811,
                         0xbdb2,
                         0x19,
                         0x57,
                         0xa0,
                         0xa6,
                         0x84,
                         0x57);

  void* query_interface(Component::uuid_t& itf_uuid) override;

  void unload() override;

  /*
   *   "pci" is in Bus:Device.Function (BDF) form. Bus and Device must be
   * zero-padded to 2 digits each, e.g. 86:00.0
   */

  /* mapped params, keys: owner,name,pci,pm_path */
  virtual Component::IKVStore* create(
      unsigned                            debug_level,
      std::map<std::string, std::string>& params) override;

  virtual IKVStore* create(const std::string& owner,
                           const std::string& param,
                           const std::string& param2) override
  {
    std::map<std::string, std::string> params;
    params["owner"]   = owner;
    params["name"]    = param;
    params["pci"]     = param2;
    params["pm_path"] = "/mnt/pmem0/";
    return create(0, params);
  }
};

#endif
