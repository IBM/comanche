#ifndef __DAWN_POOL_MANAGER_H__
#define __DAWN_POOL_MANAGER_H__

#include <api/kvstore_itf.h>
#include <map>
#include "fabric_connection_base.h"

namespace Dawn
{
using Connection_base = Fabric_connection_base;

/**
   Pool_manager tracks open pool handles on a per-shard (single thread) basis
 */
class Pool_manager {
 public:
  using pool_t = Component::IKVStore::pool_t;

  Pool_manager() {}

  /**
   * Determine if pool is open and valid
   *
   * @param pool Pool path
   */
  bool check_for_open_pool(const std::string& path, pool_t& out_pool) {
    auto i = _name_map.find(path);
    if (i == _name_map.end()) {
      PLOG("check_for_open_pool (%s) false", path.c_str());
      return false;
    }
    PLOG("check_for_open_pool (%s) true", path.c_str());
    out_pool = i->second;
    return true;
  }

  /**
   * Record pool as open
   *
   * @param pool Pool identifier
   */
  void register_pool(const std::string& path, pool_t pool) {
    assert(pool);
    if (_open_pools.find(pool) != _open_pools.end())
      throw General_exception("pool already registered");

    _open_pools[pool] = 1;
    _name_map[path] = pool;
  }

  void add_reference(pool_t pool) {
    if (_open_pools.find(pool) == _open_pools.end())
      throw Logic_exception("add reference to un-open pool");
    else
      _open_pools[pool] += 1;
  }

  /**
   * Release open pool
   *
   * @param pool Pool identifier
   *
   * @return Returns true if reference becomes 0
   */
  bool release_pool_reference(pool_t pool) {
    std::map<pool_t, unsigned>::iterator i = _open_pools.find(pool);
    if (i == _open_pools.end())
      throw Logic_exception("release_pool_reference on invalid pool");
    if (i->second == 0)
      throw Logic_exception("invalid release, reference is already");
    i->second--;
    return i->second == 0; /* return true if last reference */
  }

  /**
   *
   * Remove pool from registration, e.g. on delete
   *
   * @param pool Pool identifier
   */
  void blitz_pool_reference(pool_t pool) { _open_pools[pool] = 0; }

  /**
   * Determine if pool is open and valid
   *
   * @param pool Pool identifier
   */
  bool is_pool_open(pool_t pool) const {
    auto i = _open_pools.find(pool);
    if (i != _open_pools.end()) {
      return i->second > 0;
    }
    else
      return false;
  }

  inline const std::map<pool_t, unsigned>& open_pool_set() {
    return _open_pools;
  }

 private:
  std::map<pool_t, unsigned> _open_pools;
  std::map<std::string, pool_t> _name_map;
  std::map<pool_t, std::vector<Connection_base::memory_region_t>>
      _memory_regions;
};
}  // namespace Dawn

#endif  // __DAWN_POOL_MANAGER_H__
