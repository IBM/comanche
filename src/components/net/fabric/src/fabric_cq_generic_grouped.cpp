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

#include "fabric_cq_generic_grouped.h"

#include "async_req_record.h"
#include "fabric_comm_grouped.h"
#include "fabric_cq.h"
#include "fabric_op_control.h"
#include "fabric_runtime_error.h"
#include <boost/io/ios_state.hpp>
#include <stdexcept> /* logic_error */
#include <memory> /* unique_ptr */
#include <sstream> /* ostringstream */

class event_producer;

/**
 * Fabric/RDMA-based network component
 *
 */

/* Note: the info is owned by the caller, and must be copied if it is to be saved. */
#include <iostream>
Fabric_cq_generic_grouped::Fabric_cq_generic_grouped(
  Fabric_cq &cq_
)
  : _m_cq{}
  , _cq(cq_)
  , _m_comm_cq_set{}
  , _comm_cq_set{}
{
}

Fabric_cq_generic_grouped::~Fabric_cq_generic_grouped()
{
}

std::size_t Fabric_cq_generic_grouped::stalled_completion_count()
{
  std::lock_guard<std::mutex> k{_m_cq};
  return _cq.stalled_completion_count();
}

::fi_cq_err_entry Fabric_cq_generic_grouped::get_cq_comp_err()
{
  std::lock_guard<std::mutex> k{_m_cq};
  return _cq.get_cq_comp_err();
}

/**
 * Poll completions (e.g., completions)
 *
 * @param completion_callback (context_t, ::status_t status, void* error_data)
 *
 * @return Number of completions processed
 */

std::size_t Fabric_cq_generic_grouped::poll_completions(const Component::IFabric_op_completer::complete_old &cb_)
{
  std::size_t ct_total = 0;
  std::size_t constexpr ct_max = 1;
  Fabric_cq::fi_cq_entry_t entry;

  for ( bool drained = false; ! drained ; )
  {
    std::lock_guard<std::mutex> k{_m_cq};
    const auto ct = cq_read_locked(&entry, ct_max);
    if ( ct < 0 )
    {
      switch ( const auto e = unsigned(-ct) )
      {
      case FI_EAVAIL:
        ct_total += _cq.process_cq_comp_err(cb_);
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
    std::lock_guard<std::mutex> k0{_m_comm_cq_set};
    for ( auto &g : _comm_cq_set )
    {
      g->drain_old_completions(cb_);
    }
  }

  return ct_total;
}

std::size_t Fabric_cq_generic_grouped::poll_completions(const Component::IFabric_op_completer::complete_definite &cb_)
{
  std::size_t constexpr ct_max = 1;
  std::size_t ct_total = 0;
  Fabric_cq::fi_cq_entry_t entry;

  bool drained = false;
  while ( ! drained )
  {
    std::lock_guard<std::mutex> k{_m_cq};
    const auto ct = cq_read_locked(&entry, ct_max);
    if ( ct < 0 )
    {
      switch ( const auto e = unsigned(-ct) )
      {
      case FI_EAVAIL:
        ct_total += _cq.process_cq_comp_err(cb_);
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
    std::lock_guard<std::mutex> k0{_m_comm_cq_set};
    for ( auto &g : _comm_cq_set )
    {
      g->drain_old_completions(cb_);
    }
  }

  return ct_total;
}

std::size_t Fabric_cq_generic_grouped::poll_completions_tentative(const Component::IFabric_op_completer::complete_tentative &cb_)
{
  std::size_t constexpr ct_max = 1;
  std::size_t ct_total = 0;
  Fabric_cq::fi_cq_entry_t entry;

  bool drained = false;
  while ( ! drained )
  {
    std::lock_guard<std::mutex> k{_m_cq};
    const auto ct = cq_read_locked(&entry, ct_max);
    if ( ct < 0 )
    {
      switch ( const auto e = unsigned(-ct) )
      {
      case FI_EAVAIL:
        ct_total += _cq.process_or_queue_cq_comp_err(cb_);
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
      ct_total += _cq.process_or_queue_completion(entry, cb_, S_OK);
      g_context.release();
    }
  }

  {
    std::lock_guard<std::mutex> k0{_m_comm_cq_set};
    for ( auto &g : _comm_cq_set )
    {
      g->drain_old_completions(cb_);
    }
  }

  return ct_total;
}

std::size_t Fabric_cq_generic_grouped::poll_completions(const Component::IFabric_op_completer::complete_param_definite &cb_, void *cb_param_)
{
  std::size_t constexpr ct_max = 1;
  std::size_t ct_total = 0;
  Fabric_cq::fi_cq_entry_t entry;

  bool drained = false;
  while ( ! drained )
  {
    std::lock_guard<std::mutex> k{_m_cq};
    const auto ct = cq_read_locked(&entry, ct_max);
    if ( ct < 0 )
    {
      switch ( const auto e = unsigned(-ct) )
      {
      case FI_EAVAIL:
        ct_total += _cq.process_cq_comp_err(cb_, cb_param_);
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
      cb_(g_context->context(), S_OK, entry.flags, entry.len, nullptr, cb_param_);
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
    std::lock_guard<std::mutex> k0{_m_comm_cq_set};
    for ( auto &g : _comm_cq_set )
    {
      g->drain_old_completions(cb_, cb_param_);
    }
  }

  return ct_total;
}

std::size_t Fabric_cq_generic_grouped::poll_completions_tentative(const Component::IFabric_op_completer::complete_param_tentative &cb_, void *cb_param_)
{
  std::size_t constexpr ct_max = 1;
  std::size_t ct_total = 0;
  Fabric_cq::fi_cq_entry_t entry;

  bool drained = false;
  while ( ! drained )
  {
    std::lock_guard<std::mutex> k{_m_cq};
    const auto ct = cq_read_locked(&entry, ct_max);
    if ( ct < 0 )
    {
      switch ( const auto e = unsigned(-ct) )
      {
      case FI_EAVAIL:
        ct_total += _cq.process_or_queue_cq_comp_err(cb_, cb_param_);
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
      ct_total += _cq.process_or_queue_completion(entry, cb_, S_OK, cb_param_);
      g_context.release();
    }
  }

  {
    std::lock_guard<std::mutex> k0{_m_comm_cq_set};
    for ( auto &g : _comm_cq_set )
    {
      g->drain_old_completions(cb_, cb_param_);
    }
  }

  return ct_total;
}

void Fabric_cq_generic_grouped::member_insert(Fabric_cq_grouped *cq_)
{
  std::lock_guard<std::mutex> g{_m_comm_cq_set};
  _comm_cq_set.insert(cq_);
}

void Fabric_cq_generic_grouped::member_erase(Fabric_cq_grouped *cq_)
{
  std::lock_guard<std::mutex> g{_m_comm_cq_set};
  _comm_cq_set.erase(cq_);
}

#include <iostream>
void Fabric_cq_generic_grouped::queue_completion(Fabric_cq_grouped *cq_, ::status_t status_, const Fabric_cq::fi_cq_entry_t &cq_entry_)
{
  std::lock_guard<std::mutex> k{_m_comm_cq_set};
  auto it = _comm_cq_set.find(cq_);
  if ( it == _comm_cq_set.end() )
  {
    std::ostringstream s;
    s << "communicator " << cq_ << " not found in set of " << std::dec << _comm_cq_set.size() << " group completion queues { ";
    for ( auto jt : _comm_cq_set )
    {
      s << jt << ", ";
    }
    s << "}";
    throw std::logic_error(s.str());
  }
  (*it)->queue_completion(status_, cq_entry_);
}

ssize_t Fabric_cq_generic_grouped::cq_read(void *buf_, size_t count_) noexcept
{
  std::lock_guard<std::mutex> k{_m_cq};
  return cq_read_locked(buf_, count_);
}

ssize_t Fabric_cq_generic_grouped::cq_read_locked(void *buf_, size_t count_) noexcept
{
  return _cq.cq_read(buf_, count_);
}

ssize_t Fabric_cq_generic_grouped::cq_readerr(::fi_cq_err_entry *buf, std::uint64_t flags) noexcept
{
  std::lock_guard<std::mutex> k{_m_cq};
  return _cq.cq_readerr(buf, flags);
}
