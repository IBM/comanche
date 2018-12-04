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

#ifndef _FABRIC_CQ_H_
#define _FABRIC_CQ_H_

#include <api/fabric_itf.h> /* ::status_t */
#include "delete_copy.h" /* delete_copy */
#include "fabric_ptr.h" /* fid_unique_ptr */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fi_domain.h> /* fi_cq_attr, fi_cq_err_entry, fi_cq_data_entry, FI_CQ_FORMAT_DATA */
#pragma GCC diagnostic pop
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <queue>
#include <tuple>

struct fid_cq;

#pragma GCC diagnostic push
#if defined __GNUC__ && 6 < __GNUC__
#pragma GCC diagnostic ignored "-Wnoexcept-type"
#endif

class Fabric_cq
{
public:
  /* Largest possible format which verbs will accept. */
  static constexpr auto fi_cq_format = FI_CQ_FORMAT_DATA;
  using fi_cq_entry_t = fi_cq_data_entry;
  DELETE_COPY(Fabric_cq);
private:
  fid_unique_ptr<::fid_cq> _cq;
  const char *_type;
  std::size_t _inflight;

  using completion_t = std::tuple<::status_t, fi_cq_entry_t>;
  /* completions forwarded to client but deferred (client returned DEFER), to be retried later */
  std::queue<completion_t> _completions;

  struct stats
  {
    /* # of completions (acceptances of tentative completions only) retired by this communicator */
    std::size_t ct_total;
    /* # of deferrals (requeues) seen by this communicator */
    std::size_t defer_total;
    explicit stats()
      : ct_total{0}
      , defer_total{0}
    {
    }
    ~stats();
  } _stats;

  void queue_completion(const fi_cq_entry_t &entry, ::status_t status);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_old &completion_callback);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_definite &completion_callback);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_tentative &completion_callback);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_param_definite &completion_callback, void *callback_param);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_param_tentative &completion_callback, void *callback_param);
public:
  const char *type() { return _type; }
  explicit Fabric_cq(fid_unique_ptr<::fid_cq> &&cq, const char *type);
  ~Fabric_cq();

  /* exposed for fi_ep_bind */
  ::fid_t fid() { return &_cq->fid; }

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

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_cq_readerr fail
   */
  ::fi_cq_err_entry get_cq_comp_err();
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_cq_readerr fail
   */
  std::size_t process_cq_comp_err(const Component::IFabric_op_completer::complete_old &completion_callback);
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_cq_readerr fail
   */
  std::size_t process_cq_comp_err(const Component::IFabric_op_completer::complete_definite &completion_callback);
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_cq_readerr fail
   */
  std::size_t process_cq_comp_err(const Component::IFabric_op_completer::complete_param_definite &completion_callback, void *callback_param);
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_cq_readerr fail
   */
  std::size_t process_or_queue_cq_comp_err(const Component::IFabric_op_completer::complete_tentative &completion_callback);
  std::size_t process_or_queue_cq_comp_err(const Component::IFabric_op_completer::complete_param_tentative &completion_callback, void *callback_param);

  std::size_t process_or_queue_completion(const fi_cq_entry_t &cq_entry, const Component::IFabric_op_completer::complete_tentative &cb, ::status_t status);
  std::size_t process_or_queue_completion(const fi_cq_entry_t &cq_entry, const Component::IFabric_op_completer::complete_param_tentative &cb, ::status_t status, void *callback_param);

  ssize_t cq_read(void *buf, std::size_t count) noexcept;

  ssize_t cq_readerr(::fi_cq_err_entry *buf, std::uint64_t flags) noexcept;

  std::size_t stalled_completion_count() { return 0U; }
  void incr_inflight(const char *) { ++_inflight; }
};

#pragma GCC diagnostic pop

#endif
