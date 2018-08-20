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

#ifndef _FABRIC_GENERIC_GROUPED_H_
#define _FABRIC_GENERIC_GROUPED_H_

#include <api/fabric_itf.h> /* Component::IFabric_active_endpoint_grouped */

#include "fabric_types.h" /* addr_ep_t */
#include "fabric_op_control.h" /* fi_cq_entry_t */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fi_domain.h> /* f1_cq_err_entry */
#pragma GCC diagnostic pop

#include <unistd.h> /* ssize_t */

#include <cstdint> /* uint{32,64}_t */
#include <mutex>
#include <set>

class Fabric_comm_grouped;
class Fabric_op_control;

class Fabric_generic_grouped
  : public Component::IFabric_active_endpoint_grouped
{
  /* All communicators in a group share this "generic group."
   * Communicators need to serialize the two items ownec by the group:
   * the connection and the set of communicators.
   */
  std::mutex _m_cnxn;
  Fabric_op_control &_cnxn;

  std::mutex _m_comms;
  std::set<Fabric_comm_grouped *> _comms;

  ssize_t cq_read_locked(void *buf, std::size_t count) noexcept;

public:
  /* Begin Component::IFabric_active_endpoint_grouped (IFabric_connection) */
  /**
   * @throw std::range_error - address already registered
   * @throw std::logic_error - inconsistent memory address tables
   */
  memory_region_t register_memory(
    const void * contig_addr
    , std::size_t size
    , std::uint64_t key
    , std::uint64_t flags
  ) override;
  /**
   * @throw std::range_error - address not registered
   * @throw std::logic_error - inconsistent memory address tables
   */
  void deregister_memory(
    const memory_region_t memory_region
  ) override;
  std::uint64_t get_memory_remote_key(
    const memory_region_t memory_region
  ) const noexcept override;
  void *get_memory_descriptor(
    const memory_region_t memory_region
  ) const noexcept override;

  std::string get_peer_addr() override;
  std::string get_local_addr() override;

  std::size_t max_message_size() const noexcept override;
  /* END Component::IFabric_active_endpoint_grouped (IFabric_connection) */

  Component::IFabric_communicator *allocate_group() override;

  explicit Fabric_generic_grouped(
    Fabric_op_control &cnxn
  );

  ~Fabric_generic_grouped();

  /* BEGIN IFabric_active_endpoint_grouped (IFabric_op_completer) */
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_old &callback) override;
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_definite &callback) override;
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(const Component::IFabric_op_completer::complete_tentative &completion_callback) override;
  std::size_t stalled_completion_count() override;
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_control fail
   * @throw std::system_error : pselect fail
   */
  void wait_for_next_completion(unsigned polls_limit) override;
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_control fail
   * @throw std::system_error : pselect fail
   */
  void wait_for_next_completion(std::chrono::milliseconds timeout) override;
  void unblock_completions() override;
  /* END IFabric_active_endpoint_grouped (IFabric_op_completer) */

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_sendv fail
   */
  void post_send(const ::iovec *first, const ::iovec *last, void **desc, void *context);
  void post_send(const ::iovec *first, const ::iovec *last, void *context);
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_recvv fail
   */
  void post_recv(const ::iovec *first, const ::iovec *last, void **desc, void *context);
  void post_recv(const ::iovec *first, const ::iovec *last, void *context);
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
  );
  void post_read(
    const ::iovec *first
    , const ::iovec *last
    , std::uint64_t remote_addr
    , std::uint64_t key
    , void *context
  );
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
  );
  void post_write(
    const ::iovec *first
    , const ::iovec *last
    , std::uint64_t remote_addr
    , std::uint64_t key
    , void *context
  );
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_inject fail
   */
  void inject_send(const ::iovec *first, const ::iovec *last);

  fabric_types::addr_ep_t get_name() const;

  void forget_group(Fabric_comm_grouped *);

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_cq_readerr fail
   */
  ::fi_cq_err_entry get_cq_comp_err();
  ssize_t cq_read(void *buf, std::size_t count) noexcept;
  ssize_t cq_readerr(::fi_cq_err_entry *buf, std::uint64_t flags) noexcept;
  void queue_completion(Fabric_comm_grouped *comm, ::status_t status, const Fabric_op_control::fi_cq_entry_t &cq_entry);
};

#endif
