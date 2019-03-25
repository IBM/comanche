/*
   Copyright [2017-2019] [IBM Corporation]
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

#pragma GCC diagnostic push
#if defined __GNUC__ && 6 < __GNUC__
#pragma GCC diagnostic ignored "-Wnoexcept-type"
#endif

class Fabric_server
  : public Component::IFabric_server
  , public Fabric_connection_server
{
public:
  /*
   * @throw fabric_bad_alloc : std::bad_alloc - libfabric allocation out of memory
   */
  explicit Fabric_server(Fabric &fabric, event_producer &ep, ::fi_info & info);
  ~Fabric_server();

  /* BEGIN IFabric_op_completer */
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_old &completion_callback) override
  {
    return Fabric_connection_server::poll_completions(completion_callback);
  }
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_definite &completion_callback) override
  {
    return Fabric_connection_server::poll_completions(completion_callback);
  }
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(const Component::IFabric_op_completer::complete_tentative &completion_callback) override
  {
    return Fabric_connection_server::poll_completions_tentative(completion_callback);
  }
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions(const Component::IFabric_op_completer::complete_param_definite &completion_callback, void *callback_param) override
  {
    return Fabric_connection_server::poll_completions(completion_callback, callback_param);
  }
  /*
   * @throw fabric_runtime_error : std::runtime_error - cq_read unhandled error
   * @throw std::logic_error - called on closed connection
   */
  std::size_t poll_completions_tentative(const Component::IFabric_op_completer::complete_param_tentative &completion_callback, void *callback_param) override
  {
    return Fabric_connection_server::poll_completions_tentative(completion_callback, callback_param);
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
  /* END IFabric_op_completer */

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

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_sendv fail
   */
  void post_send(
    const ::iovec *first
    , const ::iovec *last
    , void **desc
    , void *context
  ) override { return Fabric_connection_server::post_send(first, last, desc, context); }

  void post_send(
    const std::vector<::iovec>& buffers
    , void *context
  ) override { return Fabric_connection_server::post_send(&*buffers.begin(), &*buffers.end(), context); }

  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_recvv fail
   */
  void post_recv(
    const ::iovec *first
    , const ::iovec *last
    , void **desc
    , void *context
  ) override { return Fabric_connection_server::post_recv(first, last, desc, context); }

  void post_recv(
    const std::vector<::iovec>& buffers
    , void *context
  ) override { return Fabric_op_control::post_recv(&*buffers.begin(), &*buffers.end(), context); }
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
  ) override { return Fabric_op_control::post_read(first, last, desc, remote_addr, key, context); }
  void post_read(
    const std::vector<::iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context
  ) override { return Fabric_op_control::post_read(&*buffers.begin(), &*buffers.end(), remote_addr, key, context); }
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
  ) override { return Fabric_op_control::post_write(first, last, desc, remote_addr, key, context); }
  void post_write(
    const std::vector<::iovec>& buffers,
    std::uint64_t remote_addr,
    std::uint64_t key,
    void *context
  ) override { return Fabric_op_control::post_write(&*buffers.begin(), &*buffers.end(), remote_addr, key, context); }
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_inject fail
   */
  void inject_send(
    const void *buf, std::size_t len
  ) override { return Fabric_op_control::inject_send(buf, len); }

  std::string get_peer_addr() override { return Fabric_op_control::get_peer_addr(); }
  std::string get_local_addr() override { return Fabric_op_control::get_local_addr(); }
  std::size_t max_message_size() const noexcept override { return Fabric_op_control::max_message_size(); }
  std::size_t max_inject_size() const noexcept override { return Fabric_op_control::max_inject_size(); }
};

#pragma GCC diagnostic pop

#endif
