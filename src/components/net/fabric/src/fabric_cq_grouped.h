/*
   Copyright [2017-2019] [IBM Corporation]
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


#ifndef _FABRIC_CQ_GROUPED_H_
#define _FABRIC_CQ_GROUPED_H_

#include "fabric_cq.h" /* fi_cq_entry_t */

#include <cstddef> /* size_t */
#include <mutex>
#include <queue>
#include <tuple>

class Fabric_cq_generic_grouped;
class async_req_record;

#pragma GCC diagnostic push
#if defined __GNUC__ && 6 < __GNUC__
#pragma GCC diagnostic ignored "-Wnoexcept-type"
#endif

class Fabric_cq_grouped
{
  Fabric_cq_generic_grouped &_cq;
  using completion_t = std::tuple<Fabric_cq::fi_cq_entry_t, ::status_t>;
  /* completions for this comm processed but not yet forwarded, or processed and forwarded but deferred (client returned DEFER status) */
  std::mutex _m_completions;
  std::queue<completion_t> _completions;
  struct stats
  {
    /* # of completions (acceptances of tentative completions only) retired by this communicator */
    std::size_t ct_total;
    /* # of deferrals (requeues) seen by this communicator */
    std::size_t defer_total;
    /* # of redirections of tentative completions processed by this communicator */
    std::size_t redirect_total;
    explicit stats()
      : ct_total{0}
      , defer_total{0}
      , redirect_total{0}
    {
    }
    ~stats();
  } _stats;

  std::size_t process_cq_comp_err(const Component::IFabric_op_completer::complete_old &completion_callback);
  std::size_t process_cq_comp_err(const Component::IFabric_op_completer::complete_definite &completion_callback);
  std::size_t process_cq_comp_err(const Component::IFabric_op_completer::complete_tentative &completion_callback);
  std::size_t process_cq_comp_err(const Component::IFabric_op_completer::complete_param_definite &completion_callback, void *callback_param);
  std::size_t process_cq_comp_err(const Component::IFabric_op_completer::complete_param_tentative &completion_callback, void *callback_param);
  std::size_t process_or_queue_completion(const Fabric_cq::fi_cq_entry_t &cq_entry, const Component::IFabric_op_completer::complete_old &cb, ::status_t status);
  std::size_t process_or_queue_completion(const Fabric_cq::fi_cq_entry_t &cq_entry, const Component::IFabric_op_completer::complete_definite &cb, ::status_t status);
  std::size_t process_or_queue_completion(const Fabric_cq::fi_cq_entry_t &cq_entry, const Component::IFabric_op_completer::complete_tentative &cb, ::status_t status);
  std::size_t process_or_queue_completion(const Fabric_cq::fi_cq_entry_t &cq_entry, const Component::IFabric_op_completer::complete_param_definite &cb, ::status_t status, void *callback_param);
  std::size_t process_or_queue_completion(const Fabric_cq::fi_cq_entry_t &cq_entry, const Component::IFabric_op_completer::complete_param_tentative &cb, ::status_t status, void *callback_param);
public:
  explicit Fabric_cq_grouped(Fabric_cq_generic_grouped &);
  ~Fabric_cq_grouped(); /* Note: need to notify the polling thread that this connection is going away, */

  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_old &completion_callback);
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_definite &completion_callback);
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(const Component::IFabric_op_completer::complete_tentative &completion_callback);
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_param_definite &completion_callback, void *callback_param);
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(const Component::IFabric_op_completer::complete_param_tentative &completion_callback, void *callback_param);

  std::size_t stalled_completion_count();

  void queue_completion(::status_t status, const Fabric_cq::fi_cq_entry_t &cq_entry);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_old &completion_callback);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_definite &completion_callback);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_tentative &completion_callback);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_param_definite &completion_callback, void *callback_param);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_param_tentative &completion_callback, void *callback_param);
};

#pragma GCC diagnostic pop

#endif
