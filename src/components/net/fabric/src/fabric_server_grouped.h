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

#ifndef _FABRIC_CLIENT_GROUPED_H_
#define _FABRIC_CLIENT_GROUPED_H_

#include <api/fabric_itf.h> /* Component::IFabric_server_grouped */
#include "fabric_connection_server.h"

#include "fabric_generic_grouped.h"
#include "fabric_types.h" /* addr_ep_t */

#include <unistd.h> /* ssize_t */

#include <cstdint> /* uint{16,32,64}_t */
#include <functional> /* function */
#include <mutex> /* unique_lock */

struct fi_info;
struct fi_cq_err_entry;
class event_producer;
class Fabric;
class Fabric_comm_grouped;

class Fabric_server_grouped
  : public Component::IFabric_server_grouped
  , public Fabric_connection_server
  , public Component::IFabric_communicator /* for internal use */
{
  Fabric_op_control &c() { return *this; }
  Fabric_generic_grouped _g;

  /* BEGIN Component::IFabric_server_grouped (IFabric_connection) */
  memory_region_t register_memory(
    const void * contig_addr
    , std::size_t size
    , std::uint64_t key
    , std::uint64_t flags
  ) override
  {
    return Fabric_op_control::register_memory(contig_addr, size, key, flags);
  }
  void deregister_memory(
    const memory_region_t memory_region
  ) override
  {
    return Fabric_op_control::deregister_memory(memory_region);
  };
  std::uint64_t get_memory_remote_key(
    const memory_region_t memory_region
  ) override
  {
    return Fabric_op_control::get_memory_remote_key(memory_region);
  };
  std::string get_peer_addr() override { return Fabric_op_control::get_peer_addr(); }
  std::string get_local_addr() override { return Fabric_op_control::get_local_addr(); }
  /* END Component::IFabric_server_grouped (IFabric_connection) */

  Component::IFabric_communicator *allocate_group() override { return _g.allocate_group(); }

public:
  explicit Fabric_server_grouped(
    Fabric &fabric
    , event_producer &ep
    , ::fi_info & info
  );

  ~Fabric_server_grouped();

  /* BEGIN IFabric_server_grouped (IFabric_op_completer) */
  std::size_t poll_completions(std::function<void(void *context, status_t)> completion_callback) override { return Fabric_connection_server::poll_completions(completion_callback); }
  std::size_t stalled_completion_count() override { return Fabric_connection_server::stalled_completion_count(); }
  void wait_for_next_completion(unsigned polls_limit) override { return Fabric_connection_server::wait_for_next_completion(polls_limit); }
  void wait_for_next_completion(std::chrono::milliseconds timeout) override { return Fabric_connection_server::wait_for_next_completion(timeout); }
  void unblock_completions() override { return Fabric_connection_server::unblock_completions(); }
  /* END IFabric_server_grouped (IFabric_op_completer) */

  void  post_send(const std::vector<iovec>& buffers, void *context) override { return _g.post_send(buffers, context); }
  void  post_recv(const std::vector<iovec>& buffers, void *context) override { return _g.post_recv(buffers, context); }
  void post_read(
    const std::vector<iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context
  ) override { return _g.post_read(buffers, remote_addr, key, context); }
  void post_write(
    const std::vector<iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context
  ) override { return _g.post_write(buffers, remote_addr, key, context); }
  void inject_send(const std::vector<iovec>& buffers) override { return _g.inject_send(buffers); }
  fabric_types::addr_ep_t get_name() const;

  void poll_completions_for_comm(Fabric_comm_grouped *, std::function<void(void *context, status_t)> completion_callback);
  void forget_group(Fabric_comm_grouped *);

  void *get_cq_comp_err() const;
  ssize_t cq_sread(void *buf, std::size_t count, const void *cond, int timeout) noexcept;
  ssize_t cq_readerr(::fi_cq_err_entry *buf, std::uint64_t flags) const noexcept { return _g.cq_readerr(buf, flags); }
  void queue_completion(Fabric_comm_grouped *comm, void *context, status_t status);
  void expect_event(std::uint32_t) const;
};

#endif
