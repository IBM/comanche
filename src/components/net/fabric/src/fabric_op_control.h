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
#include <rdma/fi_domain.h> /* fi_cq_attr, fi_cq_err_entry */
#pragma GCC diagnostic pop

#include <unistd.h> /* ssize_t */

#include <atomic>
#include <cstdint> /* uint{64,64}_t */
#include <memory> /* shared_ptr, unique_ptr */
#include <mutex>
#include <queue>
#include <set>
#include <vector>

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
  using completion_t = std::tuple<::status_t, ::fi_cq_tagged_entry>;
  /* completions forwarded to client but deferred with DEFER status, to be retried later */
  std::mutex _m_completions;
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

  /* BEGIN Component::IFabric_op_completer */

  /* END IFabric_op_completer */

  /* BEGIN event_consumer */
  void cb(std::uint32_t event, ::fi_eq_cm_entry &entry) noexcept override;
  void err(::fi_eq_err_entry &entry) noexcept override;
  /* END event_consumer */

  std::shared_ptr<::fid_ep> make_fid_aep(::fi_info &info, void *context) const;

  fid_mr *make_fid_mr_reg_ptr(
    const void *buf
    , std::size_t len
    , std::uint64_t access
    , std::uint64_t key
    , std::uint64_t flags
  ) const;

  fid_unique_ptr<::fid_cq> make_fid_cq(::fi_cq_attr &attr, void *context) const;

public:

  const ::fi_info &ep_info() const { return *_ep_info; }
  ::fid_ep &ep() { return *_ep; }
  void ensure_event() const;
  virtual void solicit_event() const = 0;
  virtual void wait_event() const = 0;

  std::size_t poll_completions(Component::IFabric_op_completer::complete_old completion_callback) override;
  std::size_t poll_completions(Component::IFabric_op_completer::complete_definite completion_callback) override;
  std::size_t poll_completions_tentative(Component::IFabric_op_completer::complete_tentative completion_callback) override;
  std::size_t process_or_queue_completion(const ::fi_cq_tagged_entry &cq_entry, Component::IFabric_op_completer::complete_tentative cb, ::status_t status);
  std::size_t stalled_completion_count() override { return 0U; }
  void wait_for_next_completion(unsigned polls_limit) override;
  void wait_for_next_completion(std::chrono::milliseconds timeout) override;
  void unblock_completions() override;

  std::string get_peer_addr() override;
  std::string get_local_addr() override;

  void  post_send(
    const std::vector<iovec>& buffers
    , void *context
  );
  void  post_recv(
    const std::vector<iovec>& buffers
    , void *context
  );
  void post_read(
    const std::vector<iovec>& buffers
    , std::uint64_t remote_addr
    , std::uint64_t key
    , void *context
  );
  void post_write(
    const std::vector<iovec>& buffers
    , std::uint64_t remote_addr
    , std::uint64_t key
    , void *context
  );
  void inject_send(const std::vector<iovec>& buffers);

public:
  explicit Fabric_op_control(
    Fabric &fabric
    , event_producer &ev
    , ::fi_info &info
    , std::unique_ptr<Fd_control> control
    , fabric_types::addr_ep_t (*set_peer_early)(std::unique_ptr<Fd_control> control, ::fi_info &info)
  );

  ~Fabric_op_control();

  fabric_types::addr_ep_t get_name() const;

  void poll_completions_for_comm(Fabric_comm_grouped *, Component::IFabric_op_completer::complete_old completion_callback);
  void poll_completions_for_comm(Fabric_comm_grouped *, Component::IFabric_op_completer::complete_definite completion_callback);
  void poll_completions_for_comm(Fabric_comm_grouped *, Component::IFabric_op_completer::complete_tentative completion_callback);

  ::fi_cq_err_entry get_cq_comp_err() const;
  std::size_t process_cq_comp_err(Component::IFabric_op_completer::complete_old completion_callback);
  std::size_t process_cq_comp_err(Component::IFabric_op_completer::complete_definite completion_callback);
  std::size_t process_or_queue_cq_comp_err(Component::IFabric_op_completer::complete_tentative completion_callback);

  ssize_t cq_sread(void *buf, std::size_t count, const void *cond, int timeout) noexcept;
  ssize_t cq_readerr(::fi_cq_err_entry *buf, std::uint64_t flags) const noexcept;
  void queue_completion(Fabric_comm_grouped *comm, void *context, ::status_t status);
  void expect_event(std::uint32_t) const;
  bool is_shut_down() const { return _shut_down; }

  void queue_completion(const ::fi_cq_tagged_entry &entry, ::status_t status);
  std::size_t drain_old_completions(Component::IFabric_op_completer::complete_old completion_callback);
  std::size_t drain_old_completions(Component::IFabric_op_completer::complete_definite completion_callback);
  std::size_t drain_old_completions(Component::IFabric_op_completer::complete_tentative completion_callback);
  std::size_t max_message_size() const override;
};

#endif
