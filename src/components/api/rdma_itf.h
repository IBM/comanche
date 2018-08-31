/*
   Copyright [2017] [IBM Corporation]

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

#ifndef __API_RDMA_ITF__
#define __API_RDMA_ITF__

#include <functional>
#include <common/exceptions.h>
#include <api/components.h>
#include <rdma/rdma_cma.h>

namespace Component
{

/** 
 * RDMA-based network component
 * 
 */
class IRdma : public Component::IBase
{
public:
  using memory_region_t = struct ibv_mr*;
  
  DECLARE_INTERFACE_UUID(0xfbf7b335,0x9309,0x4f6b,0x8b44,0x92,0x46,0x8b,0xb5,0x6f,0x31);

  /** 
   * Connect to a waiting peer
   * 
   * @param peer_name Name of peer
   * @param port Port number
   * 
   * @return S_OK or E_FAIL
   */
  virtual status_t connect(const std::string& peer_name, int port) = 0;

  /** 
   * Wait for a connect on a specific port
   * 
   * @param port Port number
   * 
   * @return S_OK or E_FAIL
   */
  virtual status_t wait_for_connect(int port) = 0;

  /** 
   * Register buffer for RDMA
   * 
   * @param contig_addr Pointer to contiguous region
   * @param size Size of buffer in bytes
   * 
   * @return S_OK or E_FAIL
   */
  virtual struct ibv_mr * register_memory(void * contig_addr, size_t size) = 0;

  /** 
   * Post a buffer to the connection
   * 
   * @param wid Work identifier
   * @param mr0 RDMA buffer (e.g., header)
   * @param extra_mr Additional buffer (e.g., payload)
   * 
   */
  virtual void post_send(uint64_t wid, struct ibv_mr * mr0, struct ibv_mr * extra_mr = NULL) = 0;

  /** 
   * Post a buffer to receive data
   * 
   * @param wid Work identifier
   * @param mr0 RDMA buffer (from register_memory)
   *
   */
  virtual void post_recv(uint64_t wid, struct ibv_mr * mr0) = 0;

  /** 
   * Poll completions with completion function
   * 
   * @param completion_func Completion function (called for each completion)
   * 
   * @return Number of completions
   */
  virtual int poll_completions(std::function<void(uint64_t)> completion_func = 0) = 0;

  /** 
   * Block and wait for next completion.
   * 
   * @param polls_limit Maximum number of polls
   * 
   * @return Next completion id
   */
  virtual uint64_t wait_for_next_completion(unsigned polls_limit = 0) = 0;
    
  /** 
   * Disconnect from peer
   * 
   * 
   * @return S_OK or E_FAIL
   */
  virtual status_t disconnect() = 0;
};


class IRdma_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfac7b335,0x9309,0x4f6b,0x8b44,0x92,0x46,0x8b,0xb5,0x6f,0x31);

  /** 
   * Create an instance of the Rdma component
   * 
   * @param device_name Device name (e.g., mlnx5_0)
   * 
   * @return Pointer to new instance
   */
  virtual IRdma * create(const std::string& device_name) = 0;
};

} // Component

#endif // __API_RDMA_ITF__
