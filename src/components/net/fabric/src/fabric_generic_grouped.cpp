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

#include "fabric_generic_grouped.h"

#include "async_req_record.h"
#include "fabric_comm_grouped.h"
#include "fabric_op_control.h"
#include "fabric_runtime_error.h"

#include <cassert>
#include <stdexcept> /* logic_error */
#include <memory> /* unique_ptr */

class event_producer;

/**
 * Fabric/RDMA-based network component
 *
 */

/* Note: the info is owned by the caller, and must be copied if it is to be saved. */
Fabric_generic_grouped::Fabric_generic_grouped(
  Fabric_op_control &cnxn_
)
  : _cnxn(cnxn_)
  , _m_comms{}
  , _comms{}
{
}

Fabric_generic_grouped::~Fabric_generic_grouped()
{
}

auto Fabric_generic_grouped::register_memory(
  const void * contig_addr
  , std::size_t size
  , std::uint64_t key
  , std::uint64_t flags
) -> memory_region_t
{
  return cnxn().register_memory(contig_addr, size, key, flags);
}

void Fabric_generic_grouped::deregister_memory(
  const memory_region_t memory_region
)
{
  return cnxn().deregister_memory(memory_region);
}

std::uint64_t Fabric_generic_grouped::get_memory_remote_key(
  const memory_region_t memory_region
) const noexcept
{
  return cnxn().get_memory_remote_key(memory_region);
}

std::string Fabric_generic_grouped::get_peer_addr() { return cnxn().get_peer_addr(); }
std::string Fabric_generic_grouped::get_local_addr() { return cnxn().get_local_addr(); }

void Fabric_generic_grouped::post_send(
  const std::vector<iovec>& buffers_
  , void *context_
)
{
  return cnxn().post_send(buffers_, context_);
}

std::size_t Fabric_generic_grouped::stalled_completion_count()
{
  return cnxn().stalled_completion_count();
}

void Fabric_generic_grouped::wait_for_next_completion(unsigned polls_limit)
{
  return cnxn().wait_for_next_completion(polls_limit);
}

void Fabric_generic_grouped::wait_for_next_completion(std::chrono::milliseconds timeout)
{
  return cnxn().wait_for_next_completion(timeout);
}

void Fabric_generic_grouped::unblock_completions()
{
  return cnxn().unblock_completions();
}

void Fabric_generic_grouped::post_recv(
  const std::vector<iovec>& buffers_
  , void *context_
)
{
  return cnxn().post_recv(buffers_, context_);
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
void Fabric_generic_grouped::post_read(
  const std::vector<iovec>& buffers_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  return cnxn().post_read(buffers_, remote_addr_, key_, context_);
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
void Fabric_generic_grouped::post_write(
  const std::vector<iovec>& buffers_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  return cnxn().post_write(buffers_, remote_addr_, key_, context_);
}

  /**
   * Send message without completion
   *
   * @param connection Connection to inject on
   * @param buffers Buffer vector (containing regions should be registered)
   */
void Fabric_generic_grouped::inject_send(const std::vector<iovec>& buffers_)
{
  return cnxn().inject_send(buffers_);
}

::fi_cq_err_entry Fabric_generic_grouped::get_cq_comp_err() const
{
  return cnxn().get_cq_comp_err();
}

/**
 * Poll completions (e.g., completions)
 *
 * @param completion_callback (context_t, ::status_t status, void* error_data)
 *
 * @return Number of completions processed
 */

std::size_t Fabric_generic_grouped::poll_completions(Component::IFabric_op_completer::complete_old cb_)
{
  std::size_t constexpr ct_max = 1;
  std::size_t ct_total = 0;
  fi_cq_tagged_entry entry; /* We dont actually expect a tagged entry. Spefifying this to provide the largest buffer. */

  bool drained = false;
  while ( ! drained )
  {
    auto timeout = 0; /* immediate timeout */
    auto ct = cq_sread(&entry, ct_max, nullptr, timeout);
    if ( ct < 0 )
    {
      switch ( auto e = unsigned(-ct) )
      {
      case FI_EAVAIL:
        ct_total += cnxn().process_cq_comp_err(cb_);
        break;
      case FI_EAGAIN:
        drained = true;
        break;
      default:
        throw fabric_runtime_error(e, __FILE__, __LINE__);
      }
    }
    else
    {
      std::unique_ptr<async_req_record> g_context(static_cast<async_req_record *>(entry.op_context));
      cb_(g_context->context(), S_OK);
      ++ct_total;

      g_context.release();
    }
  }

  /*
   * Note: There are two reasons why a completion might end in our local "queue":
   *  (1) It was seen by another group running poll_completions, or
   *  (2) it was rejected by a client who hoped to see some other completion first.
   * In case (1) it would be reasonable to process the queued completions before
   * newer completions. But in case (2), the client will want to see later completions
   * before returning to the rejected completion.
   */
  {
    std::unique_lock<std::mutex> k0{_m_comms};
    for ( auto &g : _comms )
    {
      g->drain_old_completions(cb_);
    }
  }

  if ( cnxn().is_shut_down() && ct_total == 0 )
  {
    throw std::logic_error(std::string("Fabric_generic_grouped") + __func__ + ": Connection closed");
  }
  return ct_total;
}

std::size_t Fabric_generic_grouped::poll_completions(Component::IFabric_op_completer::complete_definite cb_)
{
  std::size_t constexpr ct_max = 1;
  std::size_t ct_total = 0;
  fi_cq_tagged_entry entry; /* We dont actually expect a tagged entry. Spefifying this to provide the largest buffer. */

  bool drained = false;
  while ( ! drained )
  {
    auto timeout = 0; /* immediate timeout */
    auto ct = cq_sread(&entry, ct_max, nullptr, timeout);
    if ( ct < 0 )
    {
      switch ( auto e = unsigned(-ct) )
      {
      case FI_EAVAIL:
        ct_total += cnxn().process_cq_comp_err(cb_);
        break;
      case FI_EAGAIN:
        drained = true;
        break;
      default:
        throw fabric_runtime_error(e, __FILE__, __LINE__);
      }
    }
    else
    {
      std::unique_ptr<async_req_record> g_context(static_cast<async_req_record *>(entry.op_context));
      cb_(g_context->context(), S_OK, entry.flags, entry.len, nullptr);
      ++ct_total;

      g_context.release();
    }
  }

  /*
   * Note: There are two reasons why a completion might end in our local "queue":
   *  (1) It was seen by another group running poll_completions, or
   *  (2) it was rejected by a client who hoped to see some other completion first.
   * In case (1) it would be reasonable to process the queued completions before
   * newer completions. But in case (2), the client will want to see later completions
   * before returning to the rejected completion.
   */
  {
    std::unique_lock<std::mutex> k0{_m_comms};
    for ( auto &g : _comms )
    {
      g->drain_old_completions(cb_);
    }
  }

  if ( cnxn().is_shut_down() && ct_total == 0 )
  {
    throw std::logic_error(std::string("Fabric_generic_grouped") + __func__ + ": Connection closed");
  }
  return ct_total;
}

std::size_t Fabric_generic_grouped::poll_completions_tentative(Component::IFabric_op_completer::complete_tentative cb_)
{
  std::size_t constexpr ct_max = 1;
  std::size_t ct_total = 0;
  fi_cq_tagged_entry entry; /* We dont actually expect a tagged entry. Spefifying this to provide the largest buffer. */

  {
    std::unique_lock<std::mutex> k0{_m_comms};
    for ( auto &g : _comms )
    {
      g->drain_old_completions(cb_);
    }
  }

  bool drained = false;
  while ( ! drained )
  {
    auto timeout = 0; /* immediate timeout */
    auto ct = cq_sread(&entry, ct_max, nullptr, timeout);
    if ( ct < 0 )
    {
      switch ( auto e = unsigned(-ct) )
      {
      case FI_EAVAIL:
        ct_total += cnxn().process_or_queue_cq_comp_err(cb_);
        break;
      case FI_EAGAIN:
        drained = true;
        break;
      default:
        throw fabric_runtime_error(e, __FILE__, __LINE__);
      }
    }
    else
    {
      std::unique_ptr<async_req_record> g_context(static_cast<async_req_record *>(entry.op_context));
      entry.op_context = g_context->context();
      ct_total += _cnxn.process_or_queue_completion(entry, cb_, S_OK);
      g_context.release();
    }
  }

  if ( cnxn().is_shut_down() && ct_total == 0 )
  {
    throw std::logic_error(std::string("Fabric_generic_grouped") + __func__ + ": Connection closed");
  }
  return ct_total;
}

auto Fabric_generic_grouped::allocate_group() -> Component::IFabric_communicator *
{
  std::lock_guard<std::mutex> g{_m_comms};
  auto comm = new Fabric_comm_grouped(*this);
  _comms.insert(comm);
  return comm;
}

void Fabric_generic_grouped::forget_group(Fabric_comm_grouped *comm_)
{
  std::lock_guard<std::mutex> g{_m_comms};
  _comms.erase(comm_);
}

void Fabric_generic_grouped::queue_completion(Fabric_comm_grouped *comm_, void *context_, ::status_t status_, const fi_cq_tagged_entry &cq_entry_)
{
  std::lock_guard<std::mutex> k{_m_comms};
  auto it = _comms.find(comm_);
  assert(it != _comms.end());
  (*it)->queue_completion(context_, status_, cq_entry_);
}

ssize_t Fabric_generic_grouped::cq_sread(void *buf_, size_t count_, const void *cond_, int timeout_) noexcept
{
  return cnxn().cq_sread(buf_, count_, cond_, timeout_);
}

ssize_t Fabric_generic_grouped::cq_readerr(::fi_cq_err_entry *buf, std::uint64_t flags) const noexcept
{
  return cnxn().cq_readerr(buf, flags);
}

std::size_t Fabric_generic_grouped::max_message_size() const noexcept
{
  return cnxn().max_message_size();
}
