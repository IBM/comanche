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

#ifndef _FABRIC_CQ_GENERIC_GROUPED_H_
#define _FABRIC_CQ_GENERIC_GROUPED_H_

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

class Fabric_cq_grouped;
class Fabric_cq;

class Fabric_cq_generic_grouped
{
  /* All communicators in a group share this "generic group."
   * Communicators need to serialize the two items owned by the group:
   * the connection and the set of communicators.
   */
  std::mutex _m_cq;
  Fabric_cq &_cq;

  /* local completions queues for communicators in the group */
  std::mutex _m_comm_cq_set;
  std::set<Fabric_cq_grouped *> _comm_cq_set;

  ssize_t cq_read_locked(void *buf, std::size_t count) noexcept;

public:
  explicit Fabric_cq_generic_grouped(
    Fabric_cq &cnxn
  );

  ~Fabric_cq_generic_grouped();

  /* BEGIN IFabric_active_endpoint_grouped (IFabric_op_completer) */
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_old &callback);
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_definite &callback);
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(const Component::IFabric_op_completer::complete_tentative &completion_callback);
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_param_definite &callback, void *callback_param);
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(const Component::IFabric_op_completer::complete_param_tentative &completion_callback, void *callback_param);
  std::size_t stalled_completion_count();

  void member_insert(Fabric_cq_grouped *cq);
  void member_erase(Fabric_cq_grouped *cq);

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_cq_readerr fail
   */
  ::fi_cq_err_entry get_cq_comp_err();
  ssize_t cq_read(void *buf, std::size_t count) noexcept;
  ssize_t cq_readerr(::fi_cq_err_entry *buf, std::uint64_t flags) noexcept;
  void queue_completion(Fabric_cq_grouped *cq, ::status_t status, const Fabric_cq::fi_cq_entry_t &cq_entry);
};

#endif
