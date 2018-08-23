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

#ifndef _FABRIC_OP_COMPLETER_H_
#define _FABRIC_OP_COMPLETER_H_

#include <api/fabric_itf.h> /* Component::IFabric_op_completer */
#include "fabric_memory_control.h"
#include "event_consumer.h"

#include "fabric_ptr.h" /* fid_unique_ptr */
#include "fabric_types.h" /* addr_ep_t */
#include "fd_pair.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fi_domain.h> /* fi_cq_attr, fi_cq_err_entry, fi_cq_data_entry */
#pragma GCC diagnostic pop

#include <unistd.h> /* ssize_t */

#include <atomic>
#include <cstdint> /* uint{64,64}_t */
#include <memory> /* shared_ptr, unique_ptr */
#include <mutex>
#include <queue>
#include <set>

struct fi_info;
struct fi_cq_err_entry;
struct fid_cq;
struct fid_ep;
struct fid_mr;
class event_producer;
class event_registration;
class Fabric;
class Fabric_comm_grouped;
class Fd_control;

class Fabric_op_control
  : public Component::IFabric_op_completer
  , public Fabric_memory_control
  , public event_consumer
{
  /* Largest possible format which verbs will accept. */
  static constexpr auto fi_cq_format = FI_CQ_FORMAT_DATA;
public:
  using fi_cq_entry_t = fi_cq_data_entry;
private:
  using completion_t = std::tuple<::status_t, fi_cq_entry_t>;
  /* completions forwarded to client but deferred with DEFER status, to be retried later */
  std::queue<completion_t> _completions;

#if CAN_USE_WAIT_SETS
  ::fi_wait_attr _wait_attr;
  fid_unique_ptr<::fid_wait> _wait_set; /* make_fid_wait(fid_fabric &fabric, fi_wait_attr &attr) */
#endif
  std::mutex _m_fd_unblock_set;
  std::set<int> _fd_unblock_set;
  /* pingpong example used separate tx and rx completion queues.
   * Not sure why; perhaps it was for accounting.
   */
  ::fi_cq_attr _cq_attr;
  fid_unique_ptr<::fid_cq> _cq;
  std::shared_ptr<::fi_info> _ep_info;
  fabric_types::addr_ep_t _peer_addr;
  std::shared_ptr<::fid_ep> _ep;

  /* Events tagged for _ep, demultiplexed from the shared event queue to this pipe.
   * Perhaps we should provide a separate event queue for every connection, but not
   * sure if hardware would support that.
   */
  Fd_pair _event_pipe;
  std::unique_ptr<event_registration> _event_registration;

  /* true after an FI_SHUTDOWN event has been observed */
  std::atomic<bool> _shut_down;

  struct stats
  {
    /* # of completions (acceptances of tentative completions only) retired by this communicator */
    std::size_t ct_total;
    /* # of deferrals (requeues) seen by this communicator */
    std::size_t defer_total;
    stats()
      : ct_total{0}
      , defer_total{0}
    {
    }
    ~stats();
  } _stats;

  /* BEGIN Component::IFabric_op_completer */

  /* END IFabric_op_completer */

  /* BEGIN event_consumer */
  /*
   * @throw std::system_error - writing event pipe
   */
  void cb(std::uint32_t event, ::fi_eq_cm_entry &entry) noexcept override;
  /*
   * @throw std::system_error - writing event pipe
   */
  void err(::fi_eq_err_entry &entry) noexcept override;
  /* END event_consumer */

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_endpoint fail (make_fid_aep)
   */
  std::shared_ptr<::fid_ep> make_fid_aep(::fi_info &info, void *context) const;

  fid_mr *make_fid_mr_reg_ptr(
    const void *buf
    , std::size_t len
    , std::uint64_t access
    , std::uint64_t key
    , std::uint64_t flags
  ) const;

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_cq_open fail (make_fid_cq)
   */
  fid_unique_ptr<::fid_cq> make_fid_cq(::fi_cq_attr &attr, void *context) const;

  void queue_completion(const fi_cq_entry_t &entry, ::status_t status);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_old &completion_callback);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_definite &completion_callback);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_tentative &completion_callback);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_param_definite &completion_callback, void *callback_param);
  std::size_t drain_old_completions(const Component::IFabric_op_completer::complete_param_tentative &completion_callback, void *callback_param);


public:
  const ::fi_info &ep_info() const { return *_ep_info; }
  ::fid_ep &ep() { return *_ep; }
  /*
   * @throw std::system_error : pselect fail
   * @throw fabric_bad_alloc : std::bad_alloc - libfabric out of memory (creating a new server)
   * @throw std::system_error - writing event pipe (normal callback)
   * @throw std::system_error - writing event pipe (readerr_eq)
   */
  void ensure_event() const;
  /**
   * @throw fabric_bad_alloc : std::bad_alloc - libfabric out of memory (creating a new server)
   * @throw std::system_error - writing event pipe (normal callback)
   * @throw std::system_error - writing event pipe (readerr_eq)
   */
  virtual void solicit_event() const = 0;
  /*
   * @throw std::system_error : pselect fail
   */
  virtual void wait_event() const = 0;

  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_old &completion_callback) override;
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_definite &completion_callback) override;
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(const Component::IFabric_op_completer::complete_tentative &completion_callback) override;
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_param_definite &completion_callback, void *callback_param) override;
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(const Component::IFabric_op_completer::complete_param_tentative &completion_callback, void *callback_param) override;
  std::size_t stalled_completion_count() override { return 0U; }
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

  std::string get_peer_addr() override;
  std::string get_local_addr() override;

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_sendv fail
   */
  void post_send(
    const ::iovec *first
    , const ::iovec *last
    , void **desc
    , void *context
  );

  void post_send(
    const ::iovec *first
    , const ::iovec *last
    , void *context
  );

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_recvv fail
   */
  void post_recv(
    const ::iovec *first
    , const ::iovec *last
    , void **desc
    , void *context
  );

  void post_recv(
    const ::iovec *first
    , const ::iovec *last
    , void *context
  );

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

public:
  /*
   * @throw fabric_bad_alloc : std::bad_alloc - out of memory
   * @throw fabric_runtime_error : std::runtime_error : ::fi_domain fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_enable fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail (event registration)
   * @throw fabric_runtime_error : std::runtime_error : ::fi_endpoint fail (make_fid_aep)
   * @throw fabric_runtime_error : std::runtime_error : ::fi_wait_open fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_cq_open fail (make_fid_cq)
   * @throw bad_dest_addr_alloc
   * @throw std::system_error (receiving fabric server name)
   * @throw std::system_error - creating event pipe fd pair
   */
  explicit Fabric_op_control(
    Fabric &fabric
    , event_producer &ev
    , ::fi_info &info
    , std::unique_ptr<Fd_control> control
    , fabric_types::addr_ep_t (*set_peer_early)(std::unique_ptr<Fd_control> control, ::fi_info &info)
  );

  ~Fabric_op_control();

  fabric_types::addr_ep_t get_name() const;

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_cq_readerr fail
   */
  ::fi_cq_err_entry get_cq_comp_err() const;
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
  ssize_t cq_readerr(::fi_cq_err_entry *buf, std::uint64_t flags) const noexcept;
  void queue_completion(Fabric_comm_grouped *comm, void *context, ::status_t status);
  /*
   * @throw std::logic_error : unexpected event
   * @throw std::system_error : read error on event pipe
   */
  void expect_event(std::uint32_t) const;
  bool is_shut_down() const { return _shut_down; }

  std::size_t max_message_size() const noexcept override;
};

#endif
