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

#ifndef _FABRIC_CLIENT_GROUPED_H_
#define _FABRIC_CLIENT_GROUPED_H_

#include <api/fabric_itf.h> /* Component::IFabric_client_grouped */
#include "fabric_connection_client.h"

#include "fabric_generic_grouped.h"
#include "fabric_op_control.h" /* fi_cq_entry_t */
#include "fabric_types.h" /* addr_ep_t */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fi_domain.h> /* fi_cq_err_entry */
#pragma GCC diagnostic pop

#include <unistd.h> /* ssize_t */

#include <cstdint> /* uint{16,32,64}_t */
#include <mutex> /* unique_lock */
#include <set>
#include <vector>

struct fi_info;
struct fi_cq_err_entry;
class event_producer;
class Fabric;
class Fabric_comm_grouped;

class Fabric_client_grouped
  : public Component::IFabric_client_grouped
  , public Fabric_connection_client
  , public Component::IFabric_communicator /* for internal use */
{
  Fabric_op_control &c() { return *this; }
  Fabric_generic_grouped _g;

  /* BEGIN Component::IFabric_client_grouped (IFabric_connection) */
  /**
   * @throw std::range_error - address already registered
   * @throw std::logic_error - inconsistent memory address tables
   */
  memory_region_t register_memory(
    const void * contig_addr
    , std::size_t size
    , std::uint64_t key
    , std::uint64_t flags
  ) override
  {
    return Fabric_op_control::register_memory(contig_addr, size, key, flags);
  }
  /**
   * @throw std::range_error - address not registered
   * @throw std::logic_error - inconsistent memory address tables
   */
  void deregister_memory(
    const memory_region_t memory_region
  ) override
  {
    return Fabric_op_control::deregister_memory(memory_region);
  };
  std::uint64_t get_memory_remote_key(
    const memory_region_t memory_region
  ) const noexcept override
  {
    return Fabric_op_control::get_memory_remote_key(memory_region);
  };
  void *get_memory_descriptor(
    const memory_region_t memory_region
  ) const noexcept override
  {
    return Fabric_op_control::get_memory_descriptor(memory_region);
  };
  std::string get_peer_addr() override { return Fabric_op_control::get_peer_addr(); }
  std::string get_local_addr() override { return Fabric_op_control::get_local_addr(); }

  /* END Component::IFabric_client_grouped (IFabric_connection) */
  Component::IFabric_communicator *allocate_group() override { return _g.allocate_group(); }

public:
  /*
   * @throw bad_dest_addr_alloc : std::bad_alloc
   * @throw fabric_runtime_error : std::runtime_error : ::fi_domain fail
   * @throw fabric_bad_alloc : std::bad_alloc - fabric allocation out of memory
   * @throw fabric_runtime_error : std::runtime_error : ::fi_connect fail
   *
   * @throw fabric_bad_alloc : std::bad_alloc - out of memory
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_enable fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail (event registration)
   *
   * @throw std::logic_error : socket initialized with a negative value (from ::socket) in Fd_control
   * @throw std::logic_error : unexpected event
   * @throw std::system_error (receiving fabric server name)
   * @throw std::system_error : pselect fail (expecting event)
   * @throw std::system_error : resolving address
   *
   * @throw std::system_error : read error on event pipe
   * @throw std::system_error : pselect fail
   * @throw std::system_error : read error on event pipe
   * @throw fabric_bad_alloc : std::bad_alloc - libfabric out of memory (creating a new server)
   * @throw std::system_error - writing event pipe (normal callback)
   * @throw std::system_error - writing event pipe (readerr_eq)
   * @throw std::system_error - receiving data on socket
   */
  explicit Fabric_client_grouped(
    Fabric &fabric
    , event_producer &ep
    , ::fi_info & info
    , const std::string & remote
    , std::uint16_t control_port
  );

  ~Fabric_client_grouped();
  void forget_group(Fabric_comm_grouped *g) { return _g.forget_group(g); }

  /* BEGIN IFabric_client_grouped (IFabric_op_completer) */
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(Component::IFabric_op_completer::complete_old completion_callback) override
  {
    return Fabric_connection_client::poll_completions(completion_callback);
  }
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(Component::IFabric_op_completer::complete_definite completion_callback) override
  {
    return Fabric_connection_client::poll_completions(completion_callback);
  }
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(Component::IFabric_op_completer::complete_tentative completion_callback) override
  {
    return Fabric_connection_client::poll_completions_tentative(completion_callback);
  }
  std::size_t stalled_completion_count() override { return Fabric_connection_client::stalled_completion_count(); }
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_control fail
   * @throw std::system_error : pselect fail
   */
  void wait_for_next_completion(unsigned polls_limit) override { return Fabric_connection_client::wait_for_next_completion(polls_limit); }
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_control fail
   * @throw std::system_error : pselect fail
   */
  void wait_for_next_completion(std::chrono::milliseconds timeout) override { return Fabric_connection_client::wait_for_next_completion(timeout); }
  void unblock_completions() override { return Fabric_connection_client::unblock_completions(); }
  /* END IFabric_client_grouped (IFabric_op_completer) */

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_sendv fail
   */
  void post_send(const ::iovec *first, const ::iovec *last, void **desc, void *context) override { return _g.post_send(first, last, desc, context); }
  void post_send(const std::vector<::iovec>& buffers, void *context) override { return _g.post_send(&*buffers.begin(), &*buffers.end(), context); }
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_recvv fail
   */
  void post_recv(const ::iovec *first, const ::iovec *last, void **desc, void *context) override { return _g.post_recv(first, last, desc, context); }
  void post_recv(const std::vector<::iovec>& buffers, void *context) override { return _g.post_recv(&*buffers.begin(), &*buffers.end(), context); }
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_readv fail
   */
  void post_read(
    const ::iovec *first
    , const ::iovec *last
    , void **desc
    , std::uint64_t remote_addr
    , std::uint64_t key
    , void *context
  ) override { return _g.post_read(first, last, desc, remote_addr, key, context); }
  void post_read(
    const std::vector<::iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context
  ) override { return _g.post_read(&*buffers.begin(), &*buffers.end(), remote_addr, key, context); }
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_writev fail
   */
  void post_write(
    const ::iovec *first
    , const ::iovec *last
    , void **desc
    , std::uint64_t remote_addr
    , std::uint64_t key
    , void *context
  ) override { return _g.post_write(first, last, desc, remote_addr, key, context); }
  void post_write(
    const std::vector<::iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context
  ) override { return _g.post_write(&*buffers.begin(), &*buffers.end(), remote_addr, key, context); }
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_inject fail
   */
  void inject_send(const ::iovec *first, const ::iovec *last) override { return _g.inject_send(first, last); }
  void inject_send(const std::vector<::iovec>& buffers) override { return _g.inject_send(&*buffers.begin(), &*buffers.end()); }

  fabric_types::addr_ep_t get_name() const { return _g.get_name(); }

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_cq_readerr fail
   */
  ::fi_cq_err_entry get_cq_comp_err() { return _g.get_cq_comp_err(); }
  ssize_t cq_read(void *buf, std::size_t count) noexcept { return _g.cq_read(buf, count); }
  ssize_t cq_readerr(::fi_cq_err_entry *buf, std::uint64_t flags) noexcept { return _g.cq_readerr(buf, flags); }
  void queue_completion(Fabric_comm_grouped *comm, ::status_t status, const Fabric_op_control::fi_cq_entry_t &cq_entry) { return _g.queue_completion(comm, status, cq_entry); }
  /*
   * @throw std::logic_error : unexpected event
   * @throw std::system_error : read error on event pipe
  */
  std::size_t max_message_size() const noexcept override { return Fabric_connection_client::max_message_size(); }
};

#endif
