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

/*
 * Authors:
 *
 */

#include "fabric_comm_grouped.h"

#include "async_req_record.h"
#include "fabric_generic_grouped.h"
#include "fabric_op_control.h" /* fi_cq_entry_t */
#include "fabric_runtime_error.h"

#include <sys/uio.h> /* struct iovec */

/**
 * Fabric/RDMA-based network component
 *
 */

/* Note: the info is owned by the caller, and must be copied if it is to be saved. */
Fabric_comm_grouped::Fabric_comm_grouped(Fabric_generic_grouped &conn_, Fabric_cq_generic_grouped &rx_, Fabric_cq_generic_grouped &tx_)
  : _conn( conn_ )
  , _rx(rx_)
  , _tx(tx_)
{
}

Fabric_comm_grouped::~Fabric_comm_grouped()
{
/* wait until all completions are reaped */
  _conn.forget_group(this);
}

/**
 * Asynchronously post a buffer to the connection
 *
 * @param connection Connection to send on
 * @param buffers Buffer vector (containing regions should be registered)
 *
 * @return Work (context) identifier
 */
void Fabric_comm_grouped::post_send(
  const ::iovec *first_
  , const ::iovec *last_
  , void **desc_
  , void *context_
)
{
  std::unique_ptr<async_req_record> gc{new async_req_record(&_tx, context_)};
  _conn.post_send(first_, last_, desc_, &*gc);
  gc.release();
}

void Fabric_comm_grouped::post_send(
  const std::vector<::iovec>& buffers_
  , void *context_
)
{
  std::unique_ptr<async_req_record> gc{new async_req_record(&_tx, context_)};
  _conn.post_send(&*buffers_.begin(), &*buffers_.end(), &*gc);
  gc.release();
}

/**
 * Asynchronously post a buffer to receive data
 *
 * @param connection Connection to post to
 * @param buffers Buffer vector (containing regions should be registered)
 *
 * @return Work (context) identifier
 */
void Fabric_comm_grouped::post_recv(
  const ::iovec *first_
  , const ::iovec *last_
  , void **desc_
  , void *context_
)
{
  std::unique_ptr<async_req_record> gc{new async_req_record(&_rx, context_)};
  _conn.post_send(first_, last_, desc_, &*gc);
  gc.release();
}

void Fabric_comm_grouped::post_recv(
  const std::vector<::iovec>& buffers_
  , void *context_
)
{
  std::unique_ptr<async_req_record> gc{new async_req_record(&_rx, context_)};
  _conn.post_recv(&*buffers_.begin(), &*buffers_.end(), &*gc);
  gc.release();
}

  /**
   * Post RDMA read operation
   *
   * @param connection Connection to read on
   * @param buffers Destination buffer vector
   * @param remote_addr Remote address
   * @param key Key for remote address
   * @param out_context
   *
   */
void Fabric_comm_grouped::post_read(
  const ::iovec *first_
  , const ::iovec *last_
  , void **desc_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  /* ask for a read to buffer */
  std::unique_ptr<async_req_record> gc{new async_req_record(&_tx, context_)};
  _conn.post_read(first_, last_, desc_, remote_addr_, key_, &*gc);
  gc.release();
}

void Fabric_comm_grouped::post_read(
  const std::vector<::iovec>& buffers_,
  uint64_t remote_addr_,
  uint64_t key_,
  void *context_
)
{
  /* ask for a read to buffer */
  std::unique_ptr<async_req_record> gc{new async_req_record(&_tx, context_)};
  _conn.post_read(&*buffers_.begin(), &*buffers_.end(), remote_addr_, key_, &*gc);
  gc.release();
}

  /**
   * Post RDMA write operation
   *
   * @param connection Connection to write to
   * @param buffers Source buffer vector
   * @param remote_addr Remote address
   * @param key Key for remote address
   * @param out_context
   *
   */
void Fabric_comm_grouped::post_write(
  const ::iovec *first_
  , const ::iovec *last_
  , void **desc_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  std::unique_ptr<async_req_record> gc{new async_req_record(&_tx, context_)};
  _conn.post_write(first_, last_, desc_, remote_addr_, key_, &*gc);
  gc.release();
}

void Fabric_comm_grouped::post_write(
  const std::vector<::iovec>& buffers_,
  uint64_t remote_addr_,
  uint64_t key_,
  void *context_
)
{
  std::unique_ptr<async_req_record> gc{new async_req_record(&_tx, context_)};
  _conn.post_write(&*buffers_.begin(), &*buffers_.end(), remote_addr_, key_, &*gc);
  gc.release();
}

  /**
   * Send message without completion
   *
   * @param connection Connection to inject on
   * @param buffers Buffer vector (containing regions should be registered)
   */
void Fabric_comm_grouped::inject_send(const ::iovec *first_, const ::iovec *last_)
{
  _conn.inject_send(first_, last_);
}
void Fabric_comm_grouped::inject_send(const std::vector<::iovec>& buffers_)
{
  _conn.inject_send(&*buffers_.begin(), &*buffers_.end());
}

std::size_t Fabric_comm_grouped::poll_completions(const Component::IFabric_op_completer::complete_old &cb_)
{
  return _rx.poll_completions(cb_) + _tx.poll_completions(cb_);
}

std::size_t Fabric_comm_grouped::poll_completions(const Component::IFabric_op_completer::complete_definite &cb_)
{
  return _rx.poll_completions(cb_) + _tx.poll_completions(cb_);
}

std::size_t Fabric_comm_grouped::poll_completions_tentative(const Component::IFabric_op_completer::complete_tentative &cb_)
{
  return _rx.poll_completions_tentative(cb_) + _tx.poll_completions_tentative(cb_);
}

std::size_t Fabric_comm_grouped::poll_completions(const Component::IFabric_op_completer::complete_param_definite &cb_, void *cb_param_)
{
  return _rx.poll_completions(cb_, cb_param_) + _tx.poll_completions(cb_, cb_param_);
}

std::size_t Fabric_comm_grouped::poll_completions_tentative(const Component::IFabric_op_completer::complete_param_tentative &cb_, void *cb_param_)
{
  return _rx.poll_completions_tentative(cb_, cb_param_) + _tx.poll_completions_tentative(cb_, cb_param_);
}

std::size_t Fabric_comm_grouped::stalled_completion_count()
{
  return _rx.stalled_completion_count() + _tx.stalled_completion_count();
}

/**
 * Block and wait for next completion.
 *
 * @param polls_limit Maximum number of polls (throws exception on exceeding limit)
 *
 * @return Next completion context
 */
void Fabric_comm_grouped::wait_for_next_completion(std::chrono::milliseconds timeout)
{
  return _conn.wait_for_next_completion(timeout);
}

void Fabric_comm_grouped::wait_for_next_completion(unsigned polls_limit)
{
  return _conn.wait_for_next_completion(polls_limit);
}

/**
 * Unblock any threads waiting on completions
 *
 */
void Fabric_comm_grouped::unblock_completions()
{
  return _conn.unblock_completions();
}
