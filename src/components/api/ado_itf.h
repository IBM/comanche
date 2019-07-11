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

#include <common/errors.h>
#include <common/types.h>
#include <component/base.h>

#include <map>
#include <string>
#include <vector>

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

  /* ADO-to-SHARD (and vice versa) protocol */
  virtual status_t bootstrap_ado() = 0;

  virtual status_t send_memory_map(uint64_t token,
                                   size_t size,
                                   void * value_vaddr) = 0;


  virtual status_t send_work_request(uint64_t work_request_key,
                                     const void * value_addr,
                                     const size_t value_len,
                                     const void * invocation_data,
                                     const size_t invocation_len) = 0;

  
  virtual status_t check_work_completions(uint64_t& work_request_key,
                                          void *& out_response, /* use ::free to release */
                                          size_t & out_response_length) = 0;

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

  using shared_memory_token_t = uint64_t; /*< token identifying shared memory for mcas module */
  
  /**
   * Launch ADO process.  This method must NOT block.
   *
   * @param filename Location of the executable
   * @param args Command line arguments to pass
   * @param shm_token Token to pass to ADO to use to map value memory into
   * process space.
   * @param value_memory_numa_zone NUMA zone to which the value memory resides
   * @param sla Placeholder for some type of SLA/QoS requirements specification.
   *
   * @return Proxy interface, with reference count 1. Use release_ref() to
   * destroy.
   */
  virtual IADO_proxy* create(const std::string&        filename,
                             std::vector<std::string>& args,
                             numa_node_t               value_memory_numa_zone,
                             SLA*                      sla = nullptr) = 0;

  /** 
   * Wait for process to exit.
   * 
   * @param ado_proxy Handle to proxy object
   * 
   * @return S_OK on success or E_BUSY.
   */
  virtual bool has_exited(IADO_proxy* ado_proxy) = 0;

  /** 
   * Shutdown ADO process
   * 
   * @param ado Interface pointer to proxy
   * 
   * @return S_OK on success
   */
  virtual status_t shutdown(IADO_proxy* ado) = 0;
};

class IADO_manager_proxy_factory : public Component::IBase {
 public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xfacfa389,0x1665,0x4e5b,0xa1b1,0x3c,0xff,0x4a,0x5e,0xe2,0x63);
  // clang-format on

  virtual IADO_manager_proxy* create(unsigned debug_level, int core) = 0;
};

class IADO_proxy_factory : public Component::IBase {
 public:
  // clang-format off
  DECLARE_INTERFACE_UUID(0xfacbb389,0x1665,0x4e5b,0xa1b1,0x3c,0xff,0x4a,0x5e,0xe2,0x63);
  // clang-format on

  virtual IADO_proxy* create(const std::string& filename,
                             std::vector<std::string>& args,
                             std::string cores,
                             int memory) = 0;
};

} // Component

#endif // __API_ADO_ITF_H__
