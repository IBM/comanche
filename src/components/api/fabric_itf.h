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

#include <component/base.h> /* Component::IBase */

#include <chrono>
#include <cstdint>
#include <functional>
#include <tuple>
#include <vector>

struct iovec;

namespace Component
{

/**
 * Fabric/RDMA-based network component
 *
 */
class IFabric_communicator
{
public:
#if 0
  /* not currently used */
  using context_t=void*;
#endif
  virtual ~IFabric_communicator() {}

  /**
   * Asynchronously post a buffer to the connection
   *
   * @param buffers Buffer vector (containing regions should be registered)
   *
   * @return Work (context) identifier
   */

  virtual void post_send(const std::vector<iovec>& buffers, void *context) = 0;

  /**
   * Asynchronously post a buffer to receive data
   *
   * @param buffers Buffer vector (containing regions should be registered)
   *
   * @return Work (context) identifier
   */
  virtual void post_recv(const std::vector<iovec>& buffers, void *context) = 0;

  /**
   * Post RDMA read operation
   *
   * @param buffers Destination buffer vector
   * @param remote_addr Remote address
   * @param key Key for remote address
   * @param out_context
   *
   */
  virtual void post_read(const std::vector<iovec>& buffers,
                         uint64_t remote_addr,
                         uint64_t key,
                         void *context) = 0;

  /**
   * Post RDMA write operation
   *
   * @param buffers Source buffer vector
   * @param remote_addr Remote address
   * @param key Key for remote address
   * @param out_context
   *
   */
  virtual void post_write(const std::vector<iovec>& buffers,
                          uint64_t remote_addr,
                          uint64_t key,
                          void *context) = 0;

  /**
   * Send message without completion
   *
   * @param connection Connection to inject on
   * @param buffers Buffer vector (containing regions should be registered)
   */
  virtual void inject_send(const std::vector<iovec>& buffers) = 0;

  /**
   * Poll completion events; service completion queues and store
   * events not belonging to group (stall them).  This method will
   * BOTH service the completion queues and service those events
   * stalled previously
   *
   * @param completion_callback (context_t, status_t status, void* error_data, IFabric_communicator *)
   *
   * @return Number of completions processed
   */
  virtual size_t poll_completions(std::function<void(void *context, status_t)> completion_callback) = 0;

  /**
   * Get count of stalled completions.
   *
   */
  virtual size_t stalled_completion_count() = 0;

  /**
   * Block and wait for next completion.
   *
   * @param polls_limit Maximum number of polls (throws exception on exceeding limit)
   *
   * @return (was next completion context. But poll_completions can retrieve that)
   */
  virtual void wait_for_next_completion(unsigned polls_limit = 0) = 0;
  virtual void wait_for_next_completion(std::chrono::milliseconds timeout) = 0;

  /* Additional TODO:
     - support for atomic RMA operations
     - support for completion and event counters
     - support for statistics collection
  */
};

/**
 * Fabric/RDMA-based network component
 *
 */
class IFabric_connection
 : public IFabric_communicator
{
public:

  using memory_region_t=std::tuple<void *, std::uint64_t>;
#if 0
  /* not currently used */
  using context_t=IFabric_communicator::context_t;
  using endpoint_t=void*;
  using connection_t=void*;
#endif

  /**
   * Register buffer for RDMA
   *
   * @param contig_addr Pointer to contiguous region
   * @param size Size of buffer in bytes
   * @param flags Flags e.g., FI_REMOTE_READ|FI_REMOTE_WRITE
   * 
   * @return Memory region handle
   */
  virtual memory_region_t register_memory(const void * contig_addr, size_t size, uint64_t key, uint64_t flags) = 0;

  /**
   * De-register memory region
   * 
   * @param memory_region Memory region to de-register
   */
  virtual void deregister_memory(memory_region_t memory_region) = 0;

  /**
   * Allocate group (for partitioned completion handling)
   *
   */
  virtual IFabric_communicator *allocate_group() = 0;

  /**
   * Unblock any threads waiting on completions
   * 
   */
  virtual void unblock_completions() = 0;

  /**
   * Get address of connected peer (taken from fi_getpeer during
   * connection instantiation).
   * 
   * 
   * @return Peer endpoint address
   */
  virtual std::string get_peer_addr() = 0;

  /**
   * Get local address of connection (taken from fi_getname during
   * connection instantiation).
   * 
   * 
   * @return Local endpoint address
   */
  virtual std::string get_local_addr() = 0;


  /* Additional TODO:
     - support for atomic RMA operations
     - support for completion and event counters
     - support for statistics collection
  */

};


/**
 * Fabric endpoint.  Instantiation of this interface normally creates
 * an active thread that uses the fabric connection manager to handle
 * connections (see man fi_cm).  On release of the interface, the
 * endpoint is destroyed.
 */
class IFabric_endpoint : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xc373d083,0xe629,0x46c9,0x86fa,0x6f,0x96,0x40,0x61,0x10,0xdf);

  /**
   * Connect to a remote server and creates local endpoint
   * 
   * @param remote_endpoint Remote endpoint designator
   * 
   */
  virtual IFabric_connection * connect(const std::string& remote_endpoint) = 0;

  /**
   * Server/accept side handling of new connections (handled by the
   * active thread) are queued so that they can be taken by a polling
   * thread an integrated into the processing loop.  This method is
   * normally invoked until NULL is returned.
   * 
   * 
   * @return New connection or NULL on no new connections.
   */
  virtual IFabric_connection * get_new_connections() = 0;
 
  /**
   * Close connection and release any associated resources
   * 
   * @param connection
   */
  virtual void close_connection(IFabric_connection * connection) = 0;

  /**
   * Used to get a vector of active connection belonging to this
   * end point.
   * 
   * @return Vector of new connections
   */
  virtual std::vector<IFabric_connection*> connections() = 0;
 
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
  virtual std::string get_provider_name() const = 0;
};

class IFabric
{
public:
  /**
   * Open a fabric endpoint.  Endpoints usually correspond with hardware resources, e.g. verbs queue pair
   * Options may not conflict with those specified for the fabric.
   *
   * @param json_configuration Configuration string in JSON
   * @return the endpoint
   */
  virtual IFabric_endpoint * open_endpoint(const std::string& json_configuration, std::uint16_t port) = 0;
  virtual IFabric_connection * open_connection(const std::string& json_configuration, const std::string& remote_endpoint, std::uint16_t port) = 0;
};

class IFabric_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfac3d083,0xe629,0x46c9,0x86fa,0x6f,0x96,0x40,0x61,0x10,0xdf);

  /**
   * Open a fabric endpoint.  Endpoints usually correspond with hardware resources, e.g. verbs queue pair
   * Options should include provider, capabilities, active thread core.
   *
   * @param json_configuration Configuration string in JSON
   * form. e.g. { "caps":["FI_MSG","FI_RMA"], "preferred_provider" : "verbs"}
   * @return 
   */
  virtual IFabric * make_fabric(const std::string& json_configuration) = 0;
};

} // Component

#endif // __API_RDMA_ITF__
