/*
   Copyright [2018] [IBM Corporation]

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

#ifndef __API_FABRIC_ITF__
#define __API_FABRIC_ITF__

#include <functional>

namespace Component
{

/** 
 * Fabric/RDMA-based network component
 * 
 */
class IFabric : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xc373d083,0xe629,0x46c9,0x86fa,0x6f,0x96,0x40,0x61,0x10,0xdf);

  using memory_region_t=void*;
  using endpoint_t=void*;
  using context_t=void*;
  using connection_t=void*;
  
  /** 
   * Connect to a waiting peer (creates local endpoint)
   * 
   * @param remote_endpoint Remote endpoint designator
   * 
   */
  virtual connection_t connect(const std::string& remote_endpoint) = 0;

  /** 
   * Wait for a connect on a specific port
   * 
   * @param port Port number
   * @param timeout Timeout in milliseconds
   * 
   * @return S_OK or E_FAIL
   */
  virtual status_t wait_for_connect(int port, unsigned long timeout) = 0;

  /** 
   * Register buffer for RDMA
   * 
   * @param contig_addr Pointer to contiguous region
   * @param size Size of buffer in bytes
   * @param flags Flags e.g., FI_REMOTE_READ|FI_REMOTE_WRITE
   * 
   * @return Memory region handle
   */
  virtual memory_region_t register_memory(const void * contig_addr, size_t size, int flags) = 0;

  /** 
   * De-register memory region
   * 
   * @param memory_region 
   */
  virtual void deregister_memory(const memory_region_t memory_region) = 0;

  /** 
   * Asynchronously post a buffer to the connection
   * 
   * @param connection Connection to send on
   * @param buffers Buffer vector (containing regions should be registered)
   * 
   * @return Work (context) identifier
   */
  virtual context_t post_send(const connection_t connection,
                              const std::vector<struct iovec>& buffers) = 0;

  /** 
   * Asynchronously post a buffer to receive data
   * 
   * @param connection Connection to post to
   * @param buffers Buffer vector (containing regions should be registered)
   * 
   * @return Work (context) identifier
   */
  virtual context_t post_recv(const connection_t connection,
                              const std::vector<struct iovec>& buffers) = 0;

  /** 
   * Post RDMA read operation
   * 
   * @param connection Connection to read on
   * @param buffers Destination buffer vector
   * @param remote_addr Remote address
   * @param key Key for remote address
   * @param out_context 
   * 
   */
  virtual void post_read(const connection_t connection,
                         const std::vector<struct iovec>& buffers,
                         uint64_t remote_addr,
                         uint64_t key,
                         context_t& out_context) = 0;

  /** 
   * Post RDMA write operation
   * 
   * @param connection Connection to write to
   * @param buffers Source buffer vector
   * @param remote_addr Remote address
   * @param key Key for remote address
   * @param out_context 
   * 
   */
  virtual void post_write(const connection_t connection,
                          const std::vector<struct iovec>& buffers,
                          uint64_t remote_addr,
                          uint64_t key,
                          context_t& out_context) = 0;

  /** 
   * Send message without completion
   * 
   * @param connection Connection to inject on
   * @param buffers Buffer vector (containing regions should be registered)
   */
  virtual void inject_send(const connection_t connection,
                           const std::vector<struct iovec>& buffers) = 0;
  
  /** 
   * Poll events (e.g., completions)
   * 
   * @param completion_callback (context_t, status_t status, void* error_data)
   * 
   * @return Number of completions processed
   */
  virtual int poll_events(std::function<void(context_t, status_t, void*)> completion_callback) = 0;

  /** 
   * Unblock any threads waiting on completions
   * 
   */
  virtual void unblock_completions() = 0;

  /** 
   * Get the maximum message size for the provider
   * 
   * @return Max message size in bytes
   */
  virtual size_t max_message_size() const = 0;

  /** 
   * Get provider name
   * 
   * 
   * @return Provider name
   */
  virtual const std::string get_provider() const = 0;
  
  /** 
   * Close connection
   * 
   */
  virtual void close(connection_t connection) = 0;

  /* Additional TODO:
     - support for completion and event counters 
     - support for statistics collection
  */
};


class IFabric_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfac3d083,0xe629,0x46c9,0x86fa,0x6f,0x96,0x40,0x61,0x10,0xdf);

  /** 
   * Open a fabric provider instance
   * 
   * @param json_configuration Configuration string in JSON
   * form. e.g. { "caps":["FI_MSG","FI_RMA"], "preferred_provider" : "verbs"}
   * @return 
   */
  virtual IFabric * open_provider(const std::string& json_configuration) = 0;
};

} // Component

#endif // __API_RDMA_ITF__
