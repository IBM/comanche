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
#ifndef __DAWN_REGION_MANAGER_H__
#define __DAWN_REGION_MANAGER_H__

#include <api/fabric_itf.h>
#include <common/utils.h>
#include <set>
#include "connection_handler.h"
#include "types.h"

namespace Dawn
{
class Region_manager {

 public:
  Region_manager(Connection* conn) : _conn(conn) {
    assert(conn);
  }

  ~Region_manager() {
    /* deregister memory regions */
    for(auto& r : _reg) {
      _conn->deregister_memory(r);
    }
  }
  
  /**
   * Register memory with network transport for direct IO.  Cache in map.
   *
   * @param target Pointer to start or region
   * @param target_len Region length in bytes
   *
   * @return Memory region handle
   */
  inline memory_region_t ondemand_register(const void* target, size_t target_len)
  {
    /* transport will now take care of repeat registrations */
    auto mr = _conn->register_memory(target, target_len, 0, 0);
    _reg.insert(mr);
    return mr;
  }

 private:
  Connection*               _conn;
  std::set<memory_region_t> _reg;
};
}  // namespace Dawn

#endif  // __DAWN_REGION_MANAGER_H__
