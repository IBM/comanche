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

#ifndef _FABRIC_COMM_H_
#define _FABRIC_COMM_H_

#include <api/fabric_itf.h>

#include "fabric_types.h" /* addr_ep_t */

#include <chrono>
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <mutex>
#include <queue>
#include <tuple>
#include <vector>

class Fabric_connection;
class async_req_record;

class Fabric_comm
  : public Component::IFabric_communicator
{
  Fabric_connection &_conn;
  using completion_t = std::tuple<void *, status_t>;
  /* completions for this comm processed but not yet forwarded */
  std::mutex _m_completions;
  std::queue<completion_t> _completions;

  std::size_t process_cq_comp_err(std::function<void(void *context, status_t st)> completion_callback);
  std::size_t process_or_queue_completion(async_req_record *g_context_, std::function<void(void *context, status_t st)> cb_, status_t status_);
public:
  explicit Fabric_comm(Fabric_connection &);
  ~Fabric_comm(); /* Note: need to notify the polling thread that this connection is going away, */

  /* BEGIN Component::IFabric_communicator */
  void post_send(const std::vector<iovec>& buffers, void *context) override;

  void post_recv(const std::vector<iovec>& buffers, void *context) override;

  void post_read(
    const std::vector<iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context) override;

  void post_write(
    const std::vector<iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context) override;

  void inject_send(const std::vector<iovec>& buffers) override;

  std::size_t poll_completions(std::function<void(void *context, status_t)> completion_callback) override;

  std::size_t stalled_completion_count() override;

  void wait_for_next_completion(unsigned polls_limit) override;
  void wait_for_next_completion(std::chrono::milliseconds timeout) override;

  void unblock_completions() override;
  /* END Component::IFabric_communicator */

  fabric_types::addr_ep_t get_name() const;

  void queue_completion(void *context, status_t status);
  std::size_t drain_old_completions(std::function<void(void *context, status_t st) noexcept> completion_callback);
};

#endif
