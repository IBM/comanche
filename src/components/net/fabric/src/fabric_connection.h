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

#include "fabric_ptr.h"
#include "fabric_types.h" /* addr_ep_t */
#include "fd_control.h"
#include "fd_pair.h"

#include <component/base.h> /* DECLARE_VERSION, DECLARE_COMPONENT_UUID */

#include <rdma/fi_domain.h>

#include <map>
#include <memory> /* shared_ptr */
#include <mutex>
#include <set>
#include <vector>

struct fi_info;
struct fid_fabric;
struct fid_eq;
struct fid_domain;
struct fi_cq_attr;
struct fid_cq;
struct fid_ep;

class Fabric_connection
  : public Component::IFabric_connection
  , public Fabric_comm
{
  Fd_control _control;
  std::shared_ptr<fi_info> _domain_info;
  std::shared_ptr<fid_domain> _domain;
#if 0
  fi_wait_attr _wait_attr;
  fid_unique_ptr<fid_wait> _wait_set; /* make_fid_wait(fid_fabric &fabric, fi_wait_attr &attr) */
#else
  fd_set _fds_read;
  fd_set _fds_write;
  fd_set _fds_except;
#endif
  std::mutex _m_fd_unblock_set;
  std::set<int> _fd_unblock_set;
  /* pingpong example used separate tx and rx completion queues.
   * Not sure why; perhaps it was for accounting.
   */
  fi_cq_attr _cq_attr;
  fid_unique_ptr<fid_cq> _cq;
  std::shared_ptr<fi_info> _ep_info;
  fabric_types::addr_ep_t _peer_addr;
  std::shared_ptr<fid_ep> _ep;

  std::mutex _m; /* protects _mr_addr_to_desc, _mr_desc_to_addr */
  using guard = std::unique_lock<std::mutex>;
  /* Map of [starts of] registered memory regions to memory descriptors. */
  std::map<const void *, void *> _mr_addr_to_desc;
  /* since fi_mr_attr_raw may not be implemented, add reverse map as well. */
  std::map<void *, const void *> _mr_desc_to_addr;
#if 0
  static void get_rx_comp(void *ctx, std::uint64_t limit);
#endif
  std::mutex _m_comms;
  std::set<Fabric_comm *> _comms;

protected:
  const fi_info &ep_info() const { return *_ep_info; }
  fid_ep &ep() { return *_ep; }

  void await_connected(fid_eq &eq) const;
public:
  Fabric_connection(
    fid_fabric &fabric
    , fid_eq &eq
    , fi_info &info
    , Fd_control &&control
    , fabric_types::addr_ep_t (*set_peer_early)(Fd_control &control, fi_info &info));

  ~Fabric_connection(); /* Note: need to notify the polling thread that this connection is going away, */

  const Fd_control &control() const { return _control; }

  memory_region_t register_memory(const void * contig_addr, size_t size, std::uint64_t key, std::uint64_t flags) override;

  void deregister_memory(const memory_region_t memory_region) override;

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

  IFabric_communicator *allocate_group() override;

  void inject_send(const std::vector<iovec>& buffers) override;

  std::size_t poll_completions(std::function<void(void *context, status_t)> completion_callback) override;

  std::size_t stalled_completion_count() override;

  void wait_for_next_completion(unsigned polls_limit) override;
  void wait_for_next_completion(std::chrono::milliseconds timeout) override;

  void unblock_completions() override;
  std::string get_peer_addr() override;

  std::string get_local_addr() override;

  fabric_types::addr_ep_t get_name() const;

  void poll_completions_for_comm(Fabric_comm *, std::function<void(void *context, status_t)> completion_callback);

  /* public for access by Fabric_comm */
  std::vector<void *> populated_desc(const std::vector<iovec> & buffers);
  std::size_t get_cq_comp_err(std::function<void(void *connection, status_t st)> completion_callback);
  ssize_t cq_sread(void *buf, size_t count, const void *cond, int timeout) noexcept;
  ssize_t cq_readerr(fi_cq_err_entry *buf, uint64_t flags) noexcept;
  void queue_completion(Fabric_comm *comm, void *context, status_t status);
};

#endif
