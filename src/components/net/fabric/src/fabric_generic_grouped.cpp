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

#include <stdexcept> /* logic_error */
#include <memory> /* unique_ptr */
#include <sstream> /* ostringstream */

class event_producer;

/**
 * Fabric/RDMA-based network component
 *
 */

/* Note: the info is owned by the caller, and must be copied if it is to be saved. */
Fabric_generic_grouped::Fabric_generic_grouped(
  Fabric_op_control &cnxn_
)
  : _m_cnxn{}
  , _cnxn(cnxn_)
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
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.register_memory(contig_addr, size, key, flags);
}

void Fabric_generic_grouped::deregister_memory(
  const memory_region_t memory_region
)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.deregister_memory(memory_region);
}

std::uint64_t Fabric_generic_grouped::get_memory_remote_key(
  const memory_region_t memory_region
) const noexcept
{
  return _cnxn.get_memory_remote_key(memory_region);
}

void *Fabric_generic_grouped::get_memory_descriptor(
  const memory_region_t memory_region
) const noexcept
{
  return _cnxn.get_memory_descriptor(memory_region);
}

std::string Fabric_generic_grouped::get_peer_addr()
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.get_peer_addr();
}
std::string Fabric_generic_grouped::get_local_addr()
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.get_local_addr();
}

void Fabric_generic_grouped::post_send(
  const ::iovec *first_
  , const ::iovec *last_
  , void **desc_
  , void *context_
)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.post_send(first_, last_, desc_, context_);
}

void Fabric_generic_grouped::post_send(
  const ::iovec *first_
  , const ::iovec *last_
  , void *context_
)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.post_send(first_, last_, context_);
}

std::size_t Fabric_generic_grouped::stalled_completion_count()
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.stalled_completion_count();
}

void Fabric_generic_grouped::wait_for_next_completion(unsigned polls_limit)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.wait_for_next_completion(polls_limit);
}

void Fabric_generic_grouped::wait_for_next_completion(std::chrono::milliseconds timeout)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.wait_for_next_completion(timeout);
}

void Fabric_generic_grouped::unblock_completions()
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.unblock_completions();
}

void Fabric_generic_grouped::post_recv(
  const ::iovec *first_
  , const ::iovec *last_
  , void **desc_
  , void *context_
)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.post_recv(first_, last_, desc_, context_);
}

void Fabric_generic_grouped::post_recv(
  const ::iovec *first_
  , const ::iovec *last_
  , void *context_
)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.post_recv(first_, last_, context_);
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
  const ::iovec *first_
  , const ::iovec *last_
  , void **desc_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.post_read(first_, last_, desc_, remote_addr_, key_, context_);
}

void Fabric_generic_grouped::post_read(
  const ::iovec *first_
  , const ::iovec *last_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.post_read(first_, last_, remote_addr_, key_, context_);
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
  const ::iovec *first_
  , const ::iovec *last_
  , void **desc_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.post_write(first_, last_, desc_, remote_addr_, key_, context_);
}

void Fabric_generic_grouped::post_write(
  const ::iovec *first_
  , const ::iovec *last_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.post_write(first_, last_, remote_addr_, key_, context_);
}

  /**
   * Send message without completion
   *
   * @param connection Connection to inject on
   * @param buffers Buffer vector (containing regions should be registered)
   */
void Fabric_generic_grouped::inject_send(const ::iovec *first_, const ::iovec *last_)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.inject_send(first_, last_);
}

::fi_cq_err_entry Fabric_generic_grouped::get_cq_comp_err()
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.get_cq_comp_err();
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
  Fabric_op_control::fi_cq_entry_t entry;

  for ( bool drained = false; ! drained ; )
  {
    constexpr auto timeout = 0; /* immediate timeout */
    std::lock_guard<std::mutex> k{_m_cnxn};
    const auto ct = cq_sread_locked(&entry, ct_max, nullptr, timeout);
    if ( ct < 0 )
    {
      switch ( const auto e = unsigned(-ct) )
      {
      case FI_EAVAIL:
        ct_total += _cnxn.process_cq_comp_err(cb_);
        break;
      case FI_EAGAIN:
        drained = true;
        break;
      case FI_EINTR:
        /* seen when profiling with gperftools */
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
   * Note: There are two reasons why a completion might end in a communicator's queue
   *  of deferred completions:
   *  (1) It was seen by another communicator in the same group but was not owned by
   *  that communicator, or
   *  (2) it was rejected by a client who hoped to see some other completion first.
   * In case (1) it would be reasonable to process the queued completions before
   * newer completions. But in case (2), the client will want to see later completions
   * before returning to the rejected completion. Because case 2 exists, we consider
   * the old completions *after* those in the libfabric completion queue. (It would
   * not be wrong to run through the old completions first, but noght make case 2 less
   * efficient.)
   */
  {
    std::lock_guard<std::mutex> k0{_m_comms};
    for ( auto &g : _comms )
    {
      g->drain_old_completions(cb_);
    }
  }

  std::lock_guard<std::mutex> k{_m_cnxn};
  if ( _cnxn.is_shut_down() && ct_total == 0 )
  {
    throw std::logic_error(std::string("Fabric_generic_grouped") + __func__ + ": Connection closed");
  }
  return ct_total;
}

std::size_t Fabric_generic_grouped::poll_completions(Component::IFabric_op_completer::complete_definite cb_)
{
  std::size_t constexpr ct_max = 1;
  std::size_t ct_total = 0;
  Fabric_op_control::fi_cq_entry_t entry;

  bool drained = false;
  while ( ! drained )
  {
    constexpr auto timeout = 0; /* immediate timeout */
    std::lock_guard<std::mutex> k{_m_cnxn};
    const auto ct = cq_sread_locked(&entry, ct_max, nullptr, timeout);
    if ( ct < 0 )
    {
      switch ( const auto e = unsigned(-ct) )
      {
      case FI_EAVAIL:
        ct_total += _cnxn.process_cq_comp_err(cb_);
        break;
      case FI_EAGAIN:
        drained = true;
        break;
      case FI_EINTR:
        /* seen when profiling with gperftools */
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
    std::lock_guard<std::mutex> k0{_m_comms};
    for ( auto &g : _comms )
    {
      g->drain_old_completions(cb_);
    }
  }

  std::lock_guard<std::mutex> k{_m_cnxn};
  if ( _cnxn.is_shut_down() && ct_total == 0 )
  {
    throw std::logic_error(std::string("Fabric_generic_grouped") + __func__ + ": Connection closed");
  }
  return ct_total;
}

std::size_t Fabric_generic_grouped::poll_completions_tentative(Component::IFabric_op_completer::complete_tentative cb_)
{
  std::size_t constexpr ct_max = 1;
  std::size_t ct_total = 0;
  Fabric_op_control::fi_cq_entry_t entry;

  bool drained = false;
  while ( ! drained )
  {
    constexpr auto timeout = 0; /* immediate timeout */
    std::lock_guard<std::mutex> k{_m_cnxn};
    const auto ct = cq_sread_locked(&entry, ct_max, nullptr, timeout);
    if ( ct < 0 )
    {
      switch ( const auto e = unsigned(-ct) )
      {
      case FI_EAVAIL:
        ct_total += _cnxn.process_or_queue_cq_comp_err(cb_);
        break;
      case FI_EAGAIN:
        drained = true;
        break;
      case FI_EINTR:
        /* seen when profiling with gperftools */
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

  {
    std::lock_guard<std::mutex> k0{_m_comms};
    for ( auto &g : _comms )
    {
      g->drain_old_completions(cb_);
    }
  }

  std::lock_guard<std::mutex> k{_m_cnxn};
  if ( _cnxn.is_shut_down() && ct_total == 0 )
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

void Fabric_generic_grouped::queue_completion(Fabric_comm_grouped *comm_, ::status_t status_, const Fabric_op_control::fi_cq_entry_t &cq_entry_)
{
  std::lock_guard<std::mutex> k{_m_comms};
  auto it = _comms.find(comm_);
  if ( it == _comms.end() )
  {
    std::ostringstream s;
    s << "communicator " << comm_ << " not found in set of " <<_comms.size() << " communicators { ";
    for ( auto jt : _comms )
    {
      s << jt << ", ";
    }
    s << "}";
    throw std::logic_error(s.str());
  }
  (*it)->queue_completion(status_, cq_entry_);
}

ssize_t Fabric_generic_grouped::cq_sread(void *buf_, size_t count_, const void *cond_, int timeout_) noexcept
{
  std::lock_guard<std::mutex> k{_m_comms};
  return cq_sread_locked(buf_, count_, cond_, timeout_);
}

ssize_t Fabric_generic_grouped::cq_sread_locked(void *buf_, size_t count_, const void *cond_, int timeout_) noexcept
{
  return _cnxn.cq_sread(buf_, count_, cond_, timeout_);
}

ssize_t Fabric_generic_grouped::cq_readerr(::fi_cq_err_entry *buf, std::uint64_t flags) noexcept
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.cq_readerr(buf, flags);
}

std::size_t Fabric_generic_grouped::max_message_size() const noexcept
{
  return _cnxn.max_message_size();
}
