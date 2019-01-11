#ifndef __DAWN_REGION_MANAGER_H__
#define __DAWN_REGION_MANAGER_H__

#include <api/fabric_itf.h>
#include <map>
#include "connection_handler.h"
#include "types.h"

namespace Dawn
{
class Region_manager {
  static constexpr bool option_DEBUG = true;

 public:
  Region_manager(Connection* conn) : _conn(conn) { assert(conn); }

  /**
   * Register memory with network transport for direct IO.  Cache in map.
   *
   * @param target Pointer to start or region
   * @param target_len Region length in bytes
   *
   * @return Memory region handle
   */
  auto ondemand_register(const void* target, size_t target_len) {
    memory_region_t region;
    auto entry = _reg.find(target);
    if (entry != _reg.end()) {
      region = entry->second;
      if (option_DEBUG) PLOG("region already registered %p len=%lu", target, target_len);
      return region;
    }
    else {
      region = _conn->register_memory(target, target_len, 0, 0);
      _reg[target] = region;
      if (option_DEBUG) PLOG("registering memory with fabric transport %p len=%lu", target, target_len);
    }
    return region;
  }

  // /**
  //  * Get registered memory region if it exists
  //  */
  // Connection_base::memory_region_t get_preregistered(pool_t pool) {
  //   auto i = _memory_regions.find(pool);
  //   if(i == _memory_regions.end()) return nullptr;
  //   assert(i->second.size() == 1);
  //   return i->second[0]; /* for the moment return the first region handle */
  // }

 private:
  Connection* _conn;
  std::map<const void*, memory_region_t> _reg;
};
}  // namespace Dawn

#endif  // __DAWN_REGION_MANAGER_H__
