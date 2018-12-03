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

#ifndef _FABRIC_CONNECTION_SERVER_H_
#define _FABRIC_CONNECTION_SERVER_H_

#include <api/fabric_itf.h> /* Component::IFabric_server */
#include "fabric_op_control.h"

struct fi_info;

#pragma GCC diagnostic push
#if defined __GNUC__ && 6 < __GNUC__
#pragma GCC diagnostic ignored "-Wnoexcept-type"
#endif

class Fabric_connection_server
  : public Fabric_op_control
{
  /* BEGIN Fabric_op_control */
  void solicit_event() const override;
  void wait_event() const override;
  /* END Fabric_op_control */
public:
  /*
   * @throw fabric_bad_alloc : std::bad_alloc - out of memory
   * @throw fabric_runtime_error : std::runtime_error : ::fi_domain fail

   * @throw fabric_bad_alloc : std::bad_alloc - out of memory
   * @throw fabric_runtime_error : std::runtime_error : ::fi_domain fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_enable fail
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail (event registration)
   */
  explicit Fabric_connection_server(Fabric &fabric, event_producer &ep, ::fi_info & info);
  ~Fabric_connection_server();
  /* BEGIN IFabric_op_control */
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_old &completion_callback) override
  {
    return Fabric_op_control::poll_completions(completion_callback);
  }
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_param_definite &completion_callback, void *callback_param) override
  {
    return Fabric_op_control::poll_completions(completion_callback, callback_param);
  }
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(const Component::IFabric_op_completer::complete_param_tentative &completion_callback, void *callback_param) override
  {
    return Fabric_op_control::poll_completions_tentative(completion_callback, callback_param);
  }
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_definite &completion_callback) override
  {
    return Fabric_op_control::poll_completions(completion_callback);
  }
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(const Component::IFabric_op_completer::complete_tentative &completion_callback) override
  {
    return Fabric_op_control::poll_completions_tentative(completion_callback);
  }
  std::size_t stalled_completion_count() override { return Fabric_op_control::stalled_completion_count(); }
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_control fail
   * @throw std::system_error : pselect fail
   */
  void wait_for_next_completion(unsigned polls_limit) override { return Fabric_op_control::wait_for_next_completion(polls_limit); };
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_control fail
   * @throw std::system_error : pselect fail
   */
  void wait_for_next_completion(std::chrono::milliseconds timeout) override { return Fabric_op_control::wait_for_next_completion(timeout); };
  void unblock_completions() override { return Fabric_op_control::unblock_completions(); };
  /* END IFabric_op_control */
  /**
   * @throw std::range_error - address already registered
   * @throw std::logic_error - inconsistent memory address tables
   */
  memory_region_t register_memory(
    const void * contig_addr
    , std::size_t size
    , std::uint64_t key
    , std::uint64_t flags
  ) override { return Fabric_memory_control::register_memory(contig_addr, size, key, flags); }
  /**
   * @throw std::range_error - address not registered
   * @throw std::logic_error - inconsistent memory address tables
   */
  void deregister_memory(
    const memory_region_t memory_region
  ) override { return Fabric_memory_control::deregister_memory(memory_region); }
  std::uint64_t get_memory_remote_key(
    const memory_region_t memory_region
  ) const noexcept override { return Fabric_memory_control::get_memory_remote_key(memory_region); }
  void *get_memory_descriptor(
    const memory_region_t memory_region
  ) const noexcept override { return Fabric_memory_control::get_memory_descriptor(memory_region); }
  std::string get_peer_addr() override { return Fabric_op_control::get_peer_addr(); }
  std::string get_local_addr() override { return Fabric_op_control::get_local_addr(); }
};

#pragma GCC diagnostic pop

#endif
