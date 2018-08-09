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

#ifndef _FABRIC_SERVER_H_
#define _FABRIC_SERVER_H_

#include <api/fabric_itf.h> /* Component::IFabric_server */
#include "fabric_connection_server.h"

struct fi_info;
class event_producer;
class Fabric;

class Fabric_server
  : public Component::IFabric_server
  , public Fabric_connection_server
{
public:
  explicit Fabric_server(Fabric &fabric, event_producer &ep, ::fi_info & info);
  ~Fabric_server();

  /* BEGIN IFabric_op_completer */
  std::size_t poll_completions(Component::IFabric_op_completer::complete_old completion_callback) override
  {
    return Fabric_connection_server::poll_completions(completion_callback);
  }
  std::size_t poll_completions(Component::IFabric_op_completer::complete_definite completion_callback) override
  {
    return Fabric_connection_server::poll_completions(completion_callback);
  }
  std::size_t poll_completions_tentative(Component::IFabric_op_completer::complete_tentative completion_callback) override
  {
    return Fabric_connection_server::poll_completions_tentative(completion_callback);
  }
  std::size_t stalled_completion_count() override { return Fabric_op_control::stalled_completion_count(); }
  void wait_for_next_completion(unsigned polls_limit) override { return Fabric_op_control::wait_for_next_completion(polls_limit); };
  void wait_for_next_completion(std::chrono::milliseconds timeout) override { return Fabric_op_control::wait_for_next_completion(timeout); };
  void unblock_completions() override { return Fabric_op_control::unblock_completions(); };
  /* END IFabric_op_completer */

  memory_region_t register_memory(
    const void * contig_addr
    , std::size_t size
    , std::uint64_t key
    , std::uint64_t flags
  ) override { return Fabric_memory_control::register_memory(contig_addr, size, key, flags); }

  void deregister_memory(
    const memory_region_t memory_region
  ) override { return Fabric_memory_control::deregister_memory(memory_region); }

  std::uint64_t get_memory_remote_key(
    const memory_region_t memory_region
  ) override { return Fabric_memory_control::get_memory_remote_key(memory_region); }


  void  post_send(
    const std::vector<iovec>& buffers
    , void *context
  ) override { return Fabric_connection_server::post_send(buffers, context); }
  void  post_recv(
    const std::vector<iovec>& buffers
    , void *context
  ) override { return Fabric_op_control::post_recv(buffers, context); }
  void post_read(
    const std::vector<iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context
  ) override { return Fabric_op_control::post_read(buffers, remote_addr, key, context); }
  void post_write(
    const std::vector<iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context
  ) override { return Fabric_op_control::post_write(buffers, remote_addr, key, context); }
  void inject_send(
    const std::vector<iovec>& buffers
  ) override { return Fabric_op_control::inject_send(buffers); }

  std::string get_peer_addr() override { return Fabric_op_control::get_peer_addr(); }
  std::string get_local_addr() override { return Fabric_op_control::get_local_addr(); }
  std::size_t max_message_size() const override { return Fabric_op_control::max_message_size(); }
};

#endif
