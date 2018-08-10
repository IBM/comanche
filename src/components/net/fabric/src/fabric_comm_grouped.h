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

#ifndef _FABRIC_COMM_GROUPED_H_
#define _FABRIC_COMM_GROUPED_H_

#include <api/fabric_itf.h> /* Component::IFabric_communicator */

#include "fabric_types.h" /* addr_ep_t */

#include <cstddef> /* size_t */
#include <mutex>
#include <queue>
#include <tuple>

class Fabric_generic_grouped;
class async_req_record;
struct fi_cq_tagged_entry;

class Fabric_comm_grouped
  : public Component::IFabric_communicator
{
  Fabric_generic_grouped &_conn;
  using completion_t = std::tuple<::fi_cq_tagged_entry, ::status_t>;
  /* completions for this comm processed but not yet forwarded, or processed and forwarded but deferred with DEFER status */
  std::mutex _m_completions;
  std::queue<completion_t> _completions;

  std::size_t process_cq_comp_err(Component::IFabric_op_completer::complete_old completion_callback);
  std::size_t process_cq_comp_err(Component::IFabric_op_completer::complete_definite completion_callback);
  std::size_t process_cq_comp_err(Component::IFabric_op_completer::complete_tentative completion_callback);
  std::size_t process_or_queue_completion(const ::fi_cq_tagged_entry &cq_entry, Component::IFabric_op_completer::complete_old cb, ::status_t status);
  std::size_t process_or_queue_completion(const ::fi_cq_tagged_entry &cq_entry, Component::IFabric_op_completer::complete_definite cb, ::status_t status);
  std::size_t process_or_queue_completion(const ::fi_cq_tagged_entry &cq_entry, Component::IFabric_op_completer::complete_tentative cb, ::status_t status);
public:
  explicit Fabric_comm_grouped(Fabric_generic_grouped &);
  ~Fabric_comm_grouped(); /* Note: need to notify the polling thread that this connection is going away, */

  /* BEGIN Component::IFabric_communicator */
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_sendv fail
   */
  void post_send(const std::vector<iovec>& buffers, void *context) override;

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_recvv fail
   */
  void post_recv(const std::vector<iovec>& buffers, void *context) override;

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_readv fail
   */
  void post_read(
    const std::vector<iovec>& buffers
    , std::uint64_t remote_addr
    , std::uint64_t key
    , void *context
  ) override;

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_writev fail
   */
  void post_write(
    const std::vector<iovec>& buffers
    , std::uint64_t remote_addr
    , std::uint64_t key
    , void *context
  ) override;

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_inject fail
   */
  void inject_send(const std::vector<iovec>& buffers) override;

  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_sread unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(Component::IFabric_op_completer::complete_old completion_callback) override;
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_sread unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(Component::IFabric_op_completer::complete_definite completion_callback) override;
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_sread unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(Component::IFabric_op_completer::complete_tentative completion_callback) override;

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
  /* END Component::IFabric_communicator */

  fabric_types::addr_ep_t get_name() const;

  void queue_completion(void *context, ::status_t status, const ::fi_cq_tagged_entry &cq_entry);
  std::size_t drain_old_completions(Component::IFabric_op_completer::complete_old completion_callback);
  std::size_t drain_old_completions(Component::IFabric_op_completer::complete_definite completion_callback);
  std::size_t drain_old_completions(Component::IFabric_op_completer::complete_tentative completion_callback);
};

#endif
