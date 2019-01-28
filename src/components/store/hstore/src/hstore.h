/*
 * (C) Copyright IBM Corporation 2018i, 2019. All rights reserved.
 *
 */

#ifndef _DAWN_HSTORE_H_
#define _DAWN_HSTORE_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <api/kvstore_itf.h>
#pragma GCC diagnostic pop
#include <map>
#include <memory>
#include <string>

class session;

class hstore : public Component::IKVStore
{
#if 0
public:
  struct tls_cache_t {
    session *recent_session;
  };
#endif

private:
  std::mutex sessions_mutex;
  using session_map = std::map<session *, std::unique_ptr<session>>;
  session_map g_sessions;
  auto locate_session(const IKVStore::pool_t pid) -> session &;
  auto move_session(const IKVStore::pool_t pid) -> std::unique_ptr<session>;

  void delete_pool(const std::string &path, const std::string &name);
public:
  /** 
   * Constructor
   * 
   */
  hstore(const std::string &owner, const std::string &name);

  /** 
   * Destructor
   * 
   */
  ~hstore();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(
    0x1f1bf8cf,0xc2eb,0x4710,0x9bf1,0x63,0xf5,0xe8,0x1a,0xcf,0xbd
  );

  void * query_interface(Component::uuid_t& itf_uuid) override {
    return
      itf_uuid == Component::IKVStore::iid()
      ? static_cast<Component::IKVStore *>(this)
      : nullptr
      ;
  }

  void unload() override {
    delete this;
  }

  auto update_by_issue_41(const pool_t pool,
                 const std::string &key,
                 const void * value,
                 std::size_t value_len,
                 void * old_value,
                 const std::size_t old_value_len) -> status_t;

public:

  /* IKVStore */
  status_t thread_safety() const override;

  pool_t create_pool(const std::string &path,
                     const std::string &name,
                     std::size_t size,
                     unsigned int flags,
                     uint64_t expected_obj_count
                     ) override;

  pool_t open_pool(const std::string &path,
                   const std::string &name,
                   unsigned int flags) override;

  void delete_pool(pool_t pid) override;

  void close_pool(pool_t pid) override;

  status_t put(pool_t pool,
               const std::string &key,
               const void * value,
               std::size_t value_len) override;

  status_t put_direct(pool_t pool,
                      const std::string& key,
                      const void * value,
                      std::size_t value_len,
                      memory_handle_t handle) override;

  status_t get(pool_t pool,
               const std::string &key,
               void*& out_value,
               std::size_t& out_value_len) override;

  status_t get_direct(pool_t pool,
                      const std::string &key,
                      void* out_value,
                      std::size_t& out_value_len,
                      Component::IKVStore::memory_handle_t handle) override;

  key_t lock(pool_t pool,
                const std::string &key,
                lock_type_t type,
                void*& out_value,
                std::size_t& out_value_len) override;

  status_t unlock(pool_t pool,
                  key_t key_handle) override;

  status_t apply(const pool_t pool,
                 const std::string& key,
                 std::function<void(void*, size_t)> functor,
                 size_t object_size,
                 bool take_lock) override;

  status_t erase(pool_t pool,
                 const std::string &key) override;

  std::size_t count(pool_t pool) override;

  status_t map(const pool_t pool,
               std::function<int(const std::string& key,
               const void * value,
               const size_t value_len)> function) override;

  void debug(pool_t pool, unsigned cmd, uint64_t arg) override;

  status_t _apply(pool_t pool,
    const std::string& key,
    std::function<void(void*,size_t)> functor,
    std::size_t offset,
    std::size_t size,
    bool take_lock);

  status_t atomic_update(
    pool_t pool,
    const std::string& key,
    const std::vector<Operation *> &op_vector,
    bool take_lock) override;

  status_t get_pool_regions(
    pool_t pool,
    std::vector<::iovec>& out_regions) override;
};

class hstore_factory : public Component::IKVStore_factory
{
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(
    0xfacbf8cf,0xc2eb,0x4710,0x9bf1,0x63,0xf5,0xe8,0x1a,0xcf,0xbd
  );

  void * query_interface(Component::uuid_t& itf_uuid) override;

  void unload() override;

  Component::IKVStore * create(const std::string &owner,
                               const std::string &name) override;

  Component::IKVStore * create(const std::string &owner,
                               const std::string &name,
                               const std::string &) override;

  Component::IKVStore * create(unsigned debug_level,
                               const std::string &owner,
                               const std::string &name,
                               const std::string &param2) override;
};

#endif
