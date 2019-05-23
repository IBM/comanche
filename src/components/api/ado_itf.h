/*
   Copyright [2019] [IBM Corporation]
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
 * Luna Xu (xuluna@ibm.com)
 *
 */

#ifndef __API_ADO_ITF_H__
#define __API_ADO_ITF_H__

#include <string>
#include <vector>
#include <map>
#include <common/errors.h>
#include <common/types.h>

namespace Component
{

class SLA; /* to be defined - placeholder only */

/** 
 * ADO interface.  This is actually a proxy interface communicating with an external process.
 * 
 */
class IADO_proxy : public Component::IBase
{
public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xbbbfa389,0x1665,0x4e5b,0xa1b1,0x3c,0xff,0x4a,0x5e,0xe2,0x63);
  // clang-format on

  using work_id_t = uint64_t; /*< work handle/identifier */
  using shared_memory_token_t = uint64_t; /*< token identifying shared memory for mcas module */
  
  enum class Op_type {
    FLATBUFFER_OPERATION, /* a method invocation in the form of a flatbuffer message */    
  };

  /** 
   * Post work item. This method must NOT block.
   * 
   * @param type Type of work
   * @param data Pointer to work descriptor (e.g., pointer to flatbuffer)
   * @param data_len Length of descriptor data in bytes
   * @param out_work_id [out] Work identifier
   * 
   * @return S_OK or E_FULL if queue is full.
   */
  virtual status_t post_work(Op_type type, void * desc, const size_t desc_len, work_id_t& out_work_id) = 0;

  /** 
   * Check for work completions.  This gets polled by the shard process.
   * This method must NOT block.
   * 
   * @param out_completions Vector of completed work items
   * @param out_remaining_count Remaining number of work items
   * 
   * @return Number of work items completed
   */
  virtual size_t check_completions(std::vector<work_id_t>& out_completions,
                                   size_t& out_remaining_count) = 0;

  
};


/** 
 * ADO manager interface.  This is actually a proxy interface communicating with an external process.
 * The ADO manager has a "global" view of the system and can coordinate / schedule resources that
 * are being consumed by the ADO processes.
 */
class IADO_manager_proxy : public Component::IBase
{
public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xaaafa389,0x1665,0x4e5b,0xa1b1,0x3c,0xff,0x4a,0x5e,0xe2,0x63);
  // clang-format on

  /** 
   * Launch ADO process.  This method must NOT block.
   * 
   * @param filename Location of the executable
   * @param args Command line arguments to pass
   * @param shm_token Token to pass to ADO to use to map value memory into process space.
   * @param value_memory_numa_zone NUMA zone to which the value memory resides
   * @param sla Placeholder for some type of SLA/QoS requirements specification.
   * 
   * @return Proxy interface, with reference count 1. Use release_ref() to destroy.
   */
  virtual IADO * create(const std::string& filename,
                        std::vector<std::string>& args,
                        shared_memory_token_t shm_token,
                        numa_node_t value_memory_numa_zone,
                        SLA * sla = nullptr) = 0;

  /** 
   * Wait for process to exit.
   * 
   * @param ado_proxy Handle to proxy object
   * 
   * @return S_OK on success or E_BUSY.
   */
  virtual bool has_exited(IADO * ado_proxy) = 0;

  /** 
   * Shutdown ADO process
   * 
   * @param ado Interface pointer to proxy
   * 
   * @return S_OK on success
   */
  virtual status_t shutdown(IADO * ado) = 0;
};


class IADO_manager_factory : public Component::IBase
{
public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xfacfa389,0x1665,0x4e5b,0xa1b1,0x3c,0xff,0x4a,0x5e,0xe2,0x63);
  // clang-format on
  
  virtual IADO_manager_proxy * create(unsigned debug_level,
                                      std::map<std::string, std::string>& params) = 0;
}
  

} // Component

#endif // __API_ADO_ITF_H__
