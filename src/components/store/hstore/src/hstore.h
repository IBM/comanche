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


#ifndef _COMANCHE_HSTORE_H_
#define _COMANCHE_HSTORE_H_

#include "hstore_config.h"

#if THREAD_SAFE_HASH == 1
/* thread-safe hash */
#include <mutex>
#else
/* not a thread-safe hash */
#include "dummy_shared_mutex.h"
#endif

#if USE_CC_HEAP == 1
#include "allocator_cc.h"
#elif USE_CC_HEAP == 2 /* offset allocator */
#include "allocator_co.h"
#elif USE_CC_HEAP == 3 /* reconstituting allocator */
#include "allocator_rc.h"
#endif

#include "hop_hash.h"

#if USE_PMEM
#include "hstore_pmem.h"
#else
#include "hstore_nupm.h"
#include "region.h"
#endif

#include "hstore_open_pool.h"
#include "persist_fixed_string.h"
#include "pstr_equal.h"
#include "pstr_hash.h"

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

template <typename T>
  class pool_manager;
class Devdax_manager;
template <typename Handle, typename Allocator, typename Table, typename Lock>
  class session;

class hstore : public Component::IKVStore
{
  #if USE_CC_HEAP == 1
  using alloc_t = allocator_cc<char, Persister>;
  #elif USE_CC_HEAP == 2
  using alloc_t = allocator_co<char, Persister>;
  #elif USE_CC_HEAP == 3
  using alloc_t = allocator_rc<char, Persister>;
  #else /* USE_CC_HEAP */
  using alloc_t = allocator_pobj_cache_aligned<char>;
  #endif /* USE_CC_HEAP */
  using dealloc_t = typename alloc_t::deallocator_type;
  using key_t = persist_fixed_string<char, dealloc_t>;
  using mapped_t = persist_fixed_string<char, dealloc_t>;
  using allocator_segment_t = alloc_t::rebind<std::pair<const key_t, mapped_t>>::other;
#if THREAD_SAFE_HASH == 1
  /* thread-safe hash */
  using hstore_shared_mutex = std::shared_timed_mutex;
  static constexpr auto thread_model = Component::IKVStore::THREAD_MODEL_MULTI_PER_POOL;
  static constexpr auto is_thread_safe = true;
#else
/* not a thread-safe hash */
  using hstore_shared_mutex = dummy::shared_mutex;
  static constexpr auto thread_model = Component::IKVStore::THREAD_MODEL_SINGLE_PER_POOL;
  static constexpr auto is_thread_safe = false;
#endif

  using table_t =
    hop_hash<
    key_t
    , mapped_t
    , pstr_hash<key_t>
    , pstr_equal<key_t>
    , allocator_segment_t
    , hstore_shared_mutex
    >;
public:
#if USE_PMEM
  using pm = hstore_pmem;
#else
  using persist_data_t = typename impl::persist_data<allocator_segment_t, table_t::value_type>;
  using pm = hstore_nupm<region<persist_data_t, heap_rc_shared>, table_t, table_t::allocator_type, lock_type_t>;
#endif
  using open_pool_t = pm::open_pool_handle;
private:
  using session_t = session<open_pool_t, alloc_t, table_t, lock_type_t>;

  using pool_manager_t = pool_manager<open_pool_t>;
  std::shared_ptr<pool_manager_t> _pool_manager;
  std::mutex _pools_mutex;
  using pools_map = std::map<open_pool_t *, std::unique_ptr<open_pool_t>>;
  pools_map _pools;
  auto locate_session(Component::IKVStore::pool_t pid) -> open_pool_t *;
  auto move_pool(IKVStore::pool_t pid) -> std::unique_ptr<open_pool_t>;

public:
  /**
   * Constructor
   *
   */
  hstore(const std::string &owner, const std::string &name, std::unique_ptr<Devdax_manager> &&mgr);

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

public:

  int thread_safety() const;

  /**
   * Check capability of component
   *
   * @param cap Capability type
   *
   * @return 1 if supported, 0 if not supported, -1 if not recognized??
   */
  int get_capability(Capability cap) const override;

  pool_t create_pool(const std::string &name,
                     std::size_t size,
                     std::uint32_t flags,
                     std::uint64_t expected_obj_count
                     ) override;

  pool_t open_pool(const std::string &name,
                   std::uint32_t flags) override;

  status_t delete_pool(const std::string &name) override;

  status_t close_pool(pool_t pid) override;

  status_t grow_pool(pool_t pool,
                             size_t increment_size,
                             size_t& reconfigured_size ) override;

  status_t put(pool_t pool,
               const std::string &key,
               const void * value,
               std::size_t value_len,
               std::uint32_t flags = FLAGS_NONE) override;

  status_t put_direct(pool_t pool,
                      const std::string& key,
                      const void * value,
                      std::size_t value_len,
                      memory_handle_t handle = HANDLE_NONE,
                      std::uint32_t flags = FLAGS_NONE) override;

  status_t get(pool_t pool,
               const std::string &key,
               void*& out_value,
               std::size_t& out_value_len) override;

  status_t get_direct(pool_t pool,
                      const std::string &key,
                      void* out_value,
                      std::size_t& out_value_len,
                      Component::IKVStore::memory_handle_t handle) override;

  status_t get_attribute(pool_t pool,
                                 Attribute attr,
                                 std::vector<uint64_t>& out_attr,
                                 const std::string* key) override;

  status_t set_attribute(const pool_t pool,
                                 Attribute attr,
                                 const std::vector<uint64_t>& value,
                                 const std::string* key) override;

  Component::IKVStore::key_t lock(pool_t pool,
                const std::string &key,
                lock_type_t type,
                void*& out_value,
                std::size_t& out_value_len) override;

  status_t unlock(pool_t pool,
                  Component::IKVStore::key_t key_handle) override;

  status_t apply(pool_t pool,
                 const std::string& key,
                 std::function<void(void*, size_t)> functor,
                 size_t object_size,
                 bool take_lock);

  status_t erase(pool_t pool,
                 const std::string &key) override;

  std::size_t count(pool_t pool) override;

  status_t map(pool_t pool,
               std::function<int(const std::string& key,
               const void * value,
               size_t value_len)> function) override;

  status_t free_memory(void * p) override;

  void debug(pool_t pool, unsigned cmd, uint64_t arg) override;

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
                               const std::string &dax_map) override;

  Component::IKVStore * create(unsigned debug_level,
                               const std::string &owner,
                               const std::string &name,
                               const std::string &dax_map) override;
};

#endif
