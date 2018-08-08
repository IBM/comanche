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

#include <component/base.h> /* Component::IBase, DECLARE_COMPONENT_UUID, DECLARE_INTERFACE_UUID */

#include <chrono>
#include <cstdint>
#include <functional>
#include <vector>

#ifndef FABRIC_H
#define FI_READ                 (1ULL << 8)
#define FI_WRITE                (1ULL << 9)
#define FI_RECV                 (1ULL << 10)
#define FI_SEND                 (1ULL << 11)
#define FI_TRANSMIT             FI_SEND
#define FI_REMOTE_READ          (1ULL << 12)
#define FI_REMOTE_WRITE         (1ULL << 13)

#define FI_MULTI_RECV           (1ULL << 16)
#define FI_REMOTE_CQ_DATA       (1ULL << 17)
#define FI_MORE                 (1ULL << 18)
#define FI_PEEK                 (1ULL << 19)
#define FI_TRIGGER              (1ULL << 20)
#define FI_FENCE                (1ULL << 21)

#define FI_COMPLETION           (1ULL << 24)
#define FI_EVENT                FI_COMPLETION
#define FI_INJECT               (1ULL << 25)
#define FI_INJECT_COMPLETE      (1ULL << 26)
#define FI_TRANSMIT_COMPLETE    (1ULL << 27)
#define FI_DELIVERY_COMPLETE    (1ULL << 28)
#define FI_AFFINITY             (1ULL << 29)
#define FI_COMMIT_COMPLETE      (1ULL << 30)

#define FI_VARIABLE_MSG         (1ULL << 48)
#define FI_RMA_PMEM             (1ULL << 49)
#define FI_SOURCE_ERR           (1ULL << 50)
#define FI_LOCAL_COMM           (1ULL << 51)
#define FI_REMOTE_COMM          (1ULL << 52)
#define FI_SHARED_AV            (1ULL << 53)
#define FI_PROV_ATTR_ONLY       (1ULL << 54)
#define FI_NUMERICHOST          (1ULL << 55)
#define FI_RMA_EVENT            (1ULL << 56)
#define FI_SOURCE               (1ULL << 57)
#define FI_NAMED_RX_CTX         (1ULL << 58)
#define FI_DIRECTED_RECV        (1ULL << 59)
#endif

struct iovec; /* definition in <sys/uio.h> */

namespace Component
{

/**
 * Fabric/RDMA-based network component
 *
 */
class IFabric_op_completer
{
public:
  virtual ~IFabric_op_completer() {}

  /*
   * Function which probably belongs to a higher layer, but ended up here.
   * Callbacks which return DEFER are placed at the end of a queue to be
   * retried after later callbacks are given a first chance to run.
   */
  enum class cb_acceptance
  {
    ACCEPT
    , DEFER
  };

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
  virtual size_t poll_completions(std::function<void(void *context, status_t) noexcept> completion_callback) = 0;
  virtual size_t poll_completions_tentative(std::function<cb_acceptance(void *context, status_t) noexcept> completion_callback) = 0;

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

  /**
   * Unblock any threads waiting on completions
   * 
   */
  virtual void unblock_completions() = 0;

  /* Additional TODO:
     - support for completion and event counters
     - support for statistics collection
  */
};

/**
 * Fabric/RDMA-based network component
 *
 */
class IFabric_communicator
  : public IFabric_op_completer
{
public:

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

  /* Additional TODO:
     - support for atomic RMA operations
     - support for statistics collection
  */
};

struct IFabric_memory_region;

class IFabric_connection
{
public:

  virtual ~IFabric_connection() {}

  using memory_region_t = IFabric_memory_region *;

  /**
   * Register buffer for RDMA
   *
   * @param contig_addr Pointer to contiguous region
   * @param size Size of buffer in bytes
   * @param key Requested key for the remote memory. Note: if the fabric provider
   *            uses the key (i.e., the fabric provider memory region attributes
   *            do not include the FI_MR_PROV_KEY bit), then the key must be
   *            unique among registered memory regsions. As this API does not
   *            expose these attributes, the only safe strategy is to assume that
   *            the key must be unique among registered memory regsions.
   * @param flags Flags e.g., FI_REMOTE_READ|FI_REMOTE_WRITE. Flag definitions are in <rdma/fabric.h>
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

  virtual std::uint64_t get_memory_remote_key(memory_region_t) = 0;

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
 * An endpoint with a communication channel. Distinguished from a connection
 * which can provide grouped communications channels, but does not of itself
 * provide communications.
 *
 */
class IFabric_active_endpoint_comm
  : public IFabric_connection
  , public IFabric_communicator
{
};


/**
 * An endpoint established as a server.
 *
 */
class IFabric_server
  : public IFabric_active_endpoint_comm
{
};


/**
 * An endpoint established as a client.
 *
 */
class IFabric_client
  : public IFabric_active_endpoint_comm
{
};


/**
 * An endpoint "grouped", without a communication channel.
 * Can provide grouped communications channels, but does not of itself
 * provide communications.
 *
 */
class IFabric_active_endpoint_grouped
  : public IFabric_connection
  , public IFabric_op_completer
{
public:

  /**
   * Allocate group (for partitioned completion handling)
   *
   */
  virtual IFabric_communicator *allocate_group() = 0;
};


/**
 * A client endpoint which cannot initiate commands but which
 * can allocate "commuicators" which can initiate commands.
 *
 */
class IFabric_client_grouped
  : public IFabric_active_endpoint_grouped
{
};


/**
 * A server endpoint which cannot initiate commands but which
 * can allocate "commuicators" which can initiate commands.
 *
 */
class IFabric_server_grouped
  : public IFabric_active_endpoint_grouped
{
};


/**
 * Fabric passive endpoint. Instantiation of this interface normally creates
 * an active thread that uses the fabric connection manager to handle
 * connections (see man fi_cm).  On release of the interface, the endpoint
 * is destroyed.
 */
class IFabric_passive_endpoint
{
public:
  virtual ~IFabric_passive_endpoint() {}

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


/**
 * Fabric passive endpoint providing servers with ordinary (not grouped)
 * communicators.
 */
class IFabric_server_factory
  : public IFabric_passive_endpoint
{
public:

  /**
   * Server/accept side handling of new connections (handled by the
   * active thread) are queued so that they can be taken by a polling
   * thread an integrated into the processing loop.  This method is
   * normally invoked until NULL is returned.
   * 
   * 
   * @return New connection, or NULL if no new connection.
   */
  virtual IFabric_server * get_new_connection() = 0;

  /**
   * Close connection and release any associated resources
   * 
   * @param connection
   */
  virtual void close_connection(IFabric_server * connection) = 0;

  /**
   * Used to get a vector of active connection belonging to this
   * end point.
   * 
   * @return Vector of active connections
   */
  virtual std::vector<IFabric_server *> connections() = 0;

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


/* Fabric passive endpoint providing servers with grouped communicators.
 */
class IFabric_server_grouped_factory
  : public IFabric_passive_endpoint
{
public:
  /**
   * Server/accept side handling of new connections (handled by the
   * active thread) are queued so that they can be taken by a polling
   * thread an integrated into the processing loop.  This method is
   * normally invoked until NULL is returned.
   * 
   * 
   * @return New connection, or NULL if no new connection.
   */
  virtual IFabric_server_grouped * get_new_connection() = 0;

  /**
   * Close connection and release any associated resources
   * 
   * @param connection
   */
  virtual void close_connection(IFabric_server_grouped * connection) = 0;

  /**
   * Used to get a vector of active connection belonging to this
   * end point.
   * 
   * @return Vector of active connections
   */
  virtual std::vector<IFabric_server_grouped *> connections() = 0;
};


class IFabric
{
public:
  DECLARE_INTERFACE_UUID(0xc373d083,0xe629,0x46c9,0x86fa,0x6f,0x96,0x40,0x61,0x10,0xdf);
  virtual ~IFabric() {}
  /**
   * Open a fabric server factory. Endpoints usually correspond with hardware resources,
   * e.g. verbs queue pair. Options may not conflict with those specified for the fabric.
   *
   * @param json_configuration Configuration string in JSON
   * @return the endpoint
   */
  virtual IFabric_server_factory * open_server_factory(const std::string& json_configuration, std::uint16_t port) = 0;
  /**
   * Open a fabric "server grouped" factory. Endpoints usually correspond with hardware resources,
   * e.g. verbs queue pair. Options may not conflict with those specified for the fabric.
   * "Server groups" are servers in which each operation is assigned to a "communication group."
   * Each "completion group" may be separately polled for completions.
   *
   * @param json_configuration Configuration string in JSON
   * @return the endpoint
   */
  virtual IFabric_server_grouped_factory * open_server_grouped_factory(const std::string& json_configuration, std::uint16_t port) = 0;
  /**
   * Open a fabric client (active endpoint) connection to a server. Active endpoints
   * usually correspond with hardware resources, e.g. verbs queue pair. Options may
   * not conflict with those specified for the fabric.
   *
   * @param json_configuration Configuration string in JSON
   * @param remote_endpoint The IP address (URL) of the server
   * @param port The IP port on the server
   * @return the endpoint
   */
  virtual IFabric_client * open_client(const std::string& json_configuration, const std::string& remote_endpoint, std::uint16_t port) = 0;
  /**
   * Open a fabric endpoint for which communications are divided into smaller entities (groups).
   * Options may not conflict with those specified for the fabric.
   *
   * @param json_configuration Configuration string in JSON
   * @param remote_endpoint The IP address (URL) of the server
   * @param port The IP port on the server
   * @return the endpoint
   */
  virtual IFabric_client_grouped * open_client_grouped(const std::string& json_configuration, const std::string& remote_endpoint, std::uint16_t port) = 0;
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
  virtual ~IFabric_factory() {}
  virtual IFabric * make_fabric(const std::string& json_configuration) = 0;
};

} // Component

#endif // __API_RDMA_ITF__
