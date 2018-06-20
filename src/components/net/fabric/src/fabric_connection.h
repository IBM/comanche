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

#ifndef _FABRIC_CONNECTION_H_
#define _FABRIC_CONNECTION_H_

#include <api/fabric_itf.h>
#include "fabric_comm.h"
#include "event_consumer.h"

#include "fabric_ptr.h" /* fid_unique_ptr */
#include "fabric_types.h" /* addr_ep_t */
#include "fd_control.h"
#include "fd_pair.h"

#include <rdma/fi_domain.h> /* fi_cq_attr, fi_cq_err_entry */

#include <sys/select.h> /* fd_set */

#include <unistd.h> /* ssize_t */

#include <atomic>
#include <cstdint> /* uint64_t */
#include <map>
#include <memory> /* shared_ptr, unique_ptr */
#include <mutex> /* unique_lock */
#include <set>
#include <vector>

struct fi_info;
struct fi_cq_attr;
struct fid_fabric;
struct fid_eq;
struct fid_domain;
struct fid_cq;
struct fid_ep;
class event_producer;
class event_registration;
class Fabric;

class Fabric_connection
  : public Component::IFabric_connection
  , public Fabric_comm
  , public event_consumer
{
  Fabric &_fabric;
  std::shared_ptr<::fi_info> _domain_info;
  std::shared_ptr<::fid_domain> _domain;
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

  std::mutex _m; /* protects _mr_addr_to_desc, _mr_desc_to_addr */
  using guard = std::unique_lock<std::mutex>;
  /* Map of [starts of] registered memory regions to memory descriptors. */
  std::map<const void *, void *> _mr_addr_to_desc;
  /* since fi_mr_attr_raw may not be implemented, add reverse map as well. */
  std::map<void *, const void *> _mr_desc_to_addr;
  std::mutex _m_comms;
  std::set<Fabric_comm *> _comms;

  /* Events tagged for _ep, demultiplexed from the shared event queue to this pipe.
   * Perhaps we should provide a separate event queue for every connection, but not
   * sure if hardware would support that.
   */
  Fd_pair _event_pipe;
  std::unique_ptr<event_registration> _event_registration;

  /* true after an FI_SHUTDOWN event has been observed */
  std::atomic<bool> _shut_down;

  /* BEGIN Component::IFabric_connection */
  memory_region_t register_memory(const void * contig_addr, std::size_t size, std::uint64_t key, std::uint64_t flags) override;
  void deregister_memory(const memory_region_t memory_region) override;
  IFabric_communicator *allocate_group() override;
  std::string get_peer_addr() override;
  std::string get_local_addr() override;
  /* END Component::IFabric_connection */

  /* BEGIN Fabric_comm */
  void  post_send(const std::vector<iovec>& buffers, void *context) override { return Fabric_comm::post_send(buffers, context); }
  void  post_recv(const std::vector<iovec>& buffers, void *context) override { return Fabric_comm::post_recv(buffers, context); }
  void post_read(
    const std::vector<iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context) override { return Fabric_comm::post_read(buffers, remote_addr, key, context); }
  void post_write(
    const std::vector<iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context) override { return Fabric_comm::post_write(buffers, remote_addr, key, context); }
  void inject_send(const std::vector<iovec>& buffers) override;
  std::size_t poll_completions(std::function<void(void *context, status_t)> completion_callback) override;
  std::size_t stalled_completion_count() override;
  void wait_for_next_completion(unsigned polls_limit) override;
  void wait_for_next_completion(std::chrono::milliseconds timeout) override;
  void unblock_completions() override;
  /* END Fabric_comm */

  /* BEGIN event_consumer */
  void cb(std::uint32_t event, ::fi_eq_cm_entry &entry) noexcept override;
  void err(::fi_eq_err_entry &entry) noexcept override;
  /* END event_consumer */

  std::size_t process_cq_comp_err(std::function<void(void *connection, status_t st)> completion_callback);

  std::shared_ptr<::fid_ep> make_fid_aep(::fi_info &info, void *context) const;

  fid_mr *make_fid_mr_reg_ptr(
    const void *buf, std::size_t len,
    std::uint64_t access, std::uint64_t key,
    std::uint64_t flags) const;

  fid_unique_ptr<::fid_cq> make_fid_cq(::fi_cq_attr &attr, void *context) const;

protected:
  const ::fi_info &ep_info() const { return *_ep_info; }
  ::fid_ep &ep() { return *_ep; }
  void ensure_event() const;
  virtual void solicit_event() const = 0;
  virtual void wait_event() const = 0;
public:
  explicit Fabric_connection(
    Fabric &fabric
    , event_producer &ev
    , ::fi_info &info
    , std::unique_ptr<Fd_control> control
    , fabric_types::addr_ep_t (*set_peer_early)(std::unique_ptr<Fd_control> control, ::fi_info &info));

  ~Fabric_connection();

  void  post_send_internal(const std::vector<iovec>& buffers, void *context);
  void  post_recv_internal(const std::vector<iovec>& buffers, void *context);
  void post_read_internal(
    const std::vector<iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context);
  void post_write_internal(
    const std::vector<iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context);

  fabric_types::addr_ep_t get_name() const;

  void poll_completions_for_comm(Fabric_comm *, std::function<void(void *context, status_t)> completion_callback);
  void forget_group(Fabric_comm *);

  /* public for access by Fabric_comm */
  std::vector<void *> populated_desc(const std::vector<iovec> & buffers);
  void *get_cq_comp_err() const;
#if 1
  ssize_t cq_sread(void *buf, std::size_t count, const void *cond, int timeout) noexcept;
#else
  ssize_t cq_read(void *buf, std::size_t count) noexcept;
#endif
  ssize_t cq_readerr(::fi_cq_err_entry *buf, std::uint64_t flags) const noexcept;
  void queue_completion(Fabric_comm *comm, void *context, status_t status);
  void expect_event(std::uint32_t) const;
};

#endif
