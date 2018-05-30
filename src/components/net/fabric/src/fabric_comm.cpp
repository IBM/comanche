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

#include "fabric_comm.h"

#include "fabric_connection.h"

#include "fabric_util.h"
#include "fabric_error.h"
#include "async_req_record.h"

/**
 * Fabric/RDMA-based network component
 *
 */

/* Note: the info is owned by the caller, and must be copied if it is to be saved. */
Fabric_comm::Fabric_comm(Fabric_connection &conn_)
  : _conn( conn_ )
  , _m_completions{}
  , _completions{}
{
}

Fabric_comm::~Fabric_comm()
{
/* wait until all completions are reaped */
}

/**
 * Asynchronously post a buffer to the connection
 *
 * @param connection Connection to send on
 * @param buffers Buffer vector (containing regions should be registered)
 *
 * @return Work (context) identifier
 */
void Fabric_comm::post_send(
  const std::vector<iovec>& buffers, void *context)
{
  std::unique_ptr<async_req_record> gc{new async_req_record(this, context)};
  _conn.post_send_internal(buffers, &*gc);
  /* If the operation was accepted, fi now owns gc. Release it. */
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
void Fabric_comm::post_recv(
  const std::vector<iovec>& buffers, void *context)
{
  std::unique_ptr<async_req_record> gc{new async_req_record(this, context)};
  _conn.post_recv_internal(buffers, &*gc);
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
void Fabric_comm::post_read(
  const std::vector<iovec>& buffers,
  uint64_t remote_addr,
  uint64_t key,
  void *context)
{
  std::unique_ptr<async_req_record> gc{new async_req_record(this, context)};
  _conn.post_read_internal(buffers, remote_addr, key, &*gc);
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
void Fabric_comm::post_write(
  const std::vector<iovec>& buffers,
  uint64_t remote_addr,
  uint64_t key,
  void *context)
{
  std::unique_ptr<async_req_record> gc{new async_req_record(this, context)};
  _conn.post_write_internal(buffers, remote_addr, key, &*gc);
  gc.release();
}

  /**
   * Send message without completion
   *
   * @param connection Connection to inject on
   * @param buffers Buffer vector (containing regions should be registered)
   */
void Fabric_comm::inject_send(const std::vector<iovec>& buffers)
{
  _conn.inject_send(buffers);
}

void Fabric_comm::queue_completion(void *context_, status_t status_)
{
  std::lock_guard<std::mutex> k2{_m_completions};
  _completions.push(completion_t(context_, status_));
}

std::size_t Fabric_comm::process_or_queue_completion(async_req_record *g_context_, std::function<void(void *context, status_t st)> cb_, status_t status_)
{
  std::size_t ct_total = 0U;
  std::unique_ptr<async_req_record> g_context(g_context_);
  if ( g_context->comm() == this )
  {
    cb_(g_context->context(), status_);
    ++ct_total;
  }
  else
  {
    _conn.queue_completion(g_context->comm(), g_context->context(), status_);
  }
  g_context.release();

  return ct_total;
}

#include <iostream>
std::size_t Fabric_comm::get_cq_comp_err(std::function<void(void *context, status_t st)> cb_)
{
  fi_cq_err_entry err{0,0,0,0,0,0,0,0,0,0,0};
  CHECKZ(_conn.cq_readerr(&err, 0));

  std::cerr << __func__ << " : "
                  << " op_context " << err.op_context
                  << " flags " << err.flags
                  << " len " << err.len
                  << " buf " << err.buf
                  << " data " << err.data
                  << " tag " << err.tag
                  << " olen " << err.olen
                  << " err " << err.err
                  << " errno " << err.prov_errno
                  << " err_data " << err.err_data
                  << " err_data_size " << err.err_data_size
        << std::endl;

  return process_or_queue_completion(static_cast<async_req_record *>(err.op_context), cb_, E_FAIL);
}

  /**
   * Poll completions (e.g., completions)
   *
   * @param completion_callback (context_t, status_t status, void* error_data)
   *
   * @return Number of completions processed
   */

std::size_t Fabric_comm::drain_old_completions(std::function<void(void *context, status_t st) noexcept> completion_callback)
{
  std::size_t ct_total = 0U;
  std::unique_lock<std::mutex> k{_m_completions};
  while ( ! _completions.empty() )
  {
    auto &c = _completions.front();
    k.unlock();
    completion_callback(c.first, c.second);
    ++ct_total;
    k.lock();
    _completions.pop();
  }
  return ct_total;
}

std::size_t Fabric_comm::poll_completions(std::function<void(void *context, status_t st) noexcept> completion_callback)
{
  auto ct_total = drain_old_completions(completion_callback);

  bool drained = false;
  while ( ! drained )
  {
    std::size_t constexpr ct_max = 1;
    fi_cq_tagged_entry entry; /* We dont actually expect a tagged entry. Spefifying this to provide the largest buffer. */
    switch ( auto ct = _conn.cq_sread(&entry, ct_max, nullptr, 0) )
    {
    case -FI_EAVAIL:
      ct_total += get_cq_comp_err(completion_callback);
      break;
    case -FI_EAGAIN:
      drained = true;
      break;
    default:
      if ( ct < 0 )
      {
        throw fabric_error(int(-ct), __FILE__, __LINE__);
      }

      ct_total += process_or_queue_completion(static_cast<async_req_record *>(entry.op_context), completion_callback, S_OK);
      break;
    }
  }

  return ct_total;
}

std::size_t Fabric_comm::stalled_completion_count()
{
  std::lock_guard<std::mutex> k{_m_completions};
  return _completions.size();
}

/**
 * Block and wait for next completion.
 *
 * @param polls_limit Maximum number of polls (throws exception on exceeding limit)
 *
 * @return Next completion context
 */
void Fabric_comm::wait_for_next_completion(std::chrono::milliseconds timeout)
{
  return _conn.wait_for_next_completion(timeout);
}

void Fabric_comm::wait_for_next_completion(unsigned polls_limit)
{
  return _conn.wait_for_next_completion(polls_limit);
}

/**
 * Unblock any threads waiting on completions
 *
 */
void Fabric_comm::unblock_completions()
{
  return _conn.unblock_completions();
}
