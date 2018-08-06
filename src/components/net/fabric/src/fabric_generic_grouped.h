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

#include <unistd.h> /* ssize_t */

#include <cstdint> /* uint{32,64}_t */
#include <mutex>
#include <set>
#include <vector>

struct fi_cq_err_entry;
class Fabric_comm_grouped;
class Fabric_op_control;

class Fabric_generic_grouped
  : public Component::IFabric_active_endpoint_grouped
{
  Fabric_op_control &_cnxn;
  std::mutex _m_comms;
  std::set<Fabric_comm_grouped *> _comms;

  /* Begin Component::IFabric_active_endpoint_grouped (IFabric_connection) */
  memory_region_t register_memory(
    const void * contig_addr
    , std::size_t size
    , std::uint64_t key
    , std::uint64_t flags
  ) override;
  void deregister_memory(
    const memory_region_t memory_region
  ) override;
  std::uint64_t get_memory_remote_key(
    const memory_region_t memory_region
  ) override;

  std::string get_peer_addr() override;
  std::string get_local_addr() override;
  /* END Component::IFabric_active_endpoint_grouped (IFabric_connection) */

public:
  Component::IFabric_communicator *allocate_group() override;
private:
  Fabric_op_control &cnxn() const { return _cnxn; }

public:
  explicit Fabric_generic_grouped(
    Fabric_op_control &cnxn
  );

  ~Fabric_generic_grouped();

  /* BEGIN IFabric_active_endpoint_grouped (IFabric_op_completer) */
  std::size_t poll_completions(std::function<void(void *context, status_t)> completion_callback) override;
  std::size_t poll_completions_tentative(std::function<cb_acceptance(void *context, status_t)> completion_callback) override;
  std::size_t stalled_completion_count() override;
  void wait_for_next_completion(unsigned polls_limit) override;
  void wait_for_next_completion(std::chrono::milliseconds timeout) override;
  void unblock_completions() override;
  /* END IFabric_active_endpoint_grouped (IFabric_op_completer) */

  void  post_send(const std::vector<iovec>& buffers, void *context);
  void  post_recv(const std::vector<iovec>& buffers, void *context);
  void post_read(
    const std::vector<iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context);
  void post_write(
    const std::vector<iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context);
  void inject_send(const std::vector<iovec>& buffers);

  fabric_types::addr_ep_t get_name() const;

  void poll_completions_for_comm(Fabric_comm_grouped *, std::function<void(void *context, status_t)> completion_callback);
  void poll_completions_for_comm(Fabric_comm_grouped *, std::function<cb_acceptance(void *context, status_t)> completion_callback);
  void forget_group(Fabric_comm_grouped *);

  void *get_cq_comp_err() const;
  ssize_t cq_sread(void *buf, std::size_t count, const void *cond, int timeout) noexcept;
  ssize_t cq_readerr(::fi_cq_err_entry *buf, std::uint64_t flags) const noexcept;
  void queue_completion(Fabric_comm_grouped *comm, void *context, status_t status);
  void expect_event(std::uint32_t) const;
};

#endif
