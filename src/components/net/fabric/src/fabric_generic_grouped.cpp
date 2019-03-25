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


/*
 * Authors:
 *
 */

#include "fabric_generic_grouped.h"

#include "async_req_record.h"
#include "fabric_comm_grouped.h"
#include "fabric_cq.h"
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
  , Fabric_cq &rxcq_
  , Fabric_cq &txcq_
)
  : _m_cnxn{}
  , _cnxn(cnxn_)
  , _m_rxcq{}
  , _rxcq(rxcq_)
  , _m_txcq{}
  , _txcq(txcq_)
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
void Fabric_generic_grouped::inject_send(const void *buf_, std::size_t len_)
{
  std::lock_guard<std::mutex> k{_m_cnxn};
  return _cnxn.inject_send(buf_, len_);
}

/**
 * Poll completions (e.g., completions)
 *
 * @param completion_callback (context_t, ::status_t status, void* error_data)
 *
 * @return Number of completions processed
 */

std::size_t Fabric_generic_grouped::poll_completions(const Component::IFabric_op_completer::complete_old &cb_)
{
  std::size_t ct_total = 0;

  {
    std::lock_guard<std::mutex> k{_m_rxcq};
    ct_total += _rxcq.poll_completions(cb_);
  }
  {
    std::lock_guard<std::mutex> k{_m_txcq};
    ct_total += _txcq.poll_completions(cb_);
  }

  std::lock_guard<std::mutex> k{_m_cnxn};
  if ( _cnxn.is_shut_down() && ct_total == 0 )
  {
    throw std::logic_error(std::string("Fabric_generic_grouped") + __func__ + ": Connection closed");
  }

  return ct_total;
}

std::size_t Fabric_generic_grouped::poll_completions(const Component::IFabric_op_completer::complete_definite &cb_)
{
  std::size_t ct_total = 0;

  {
    std::lock_guard<std::mutex> k{_m_rxcq};
    ct_total += _rxcq.poll_completions(cb_);
  }
  {
    std::lock_guard<std::mutex> k{_m_txcq};
    ct_total += _txcq.poll_completions(cb_);
  }

  std::lock_guard<std::mutex> k{_m_cnxn};
  if ( _cnxn.is_shut_down() && ct_total == 0 )
  {
    throw std::logic_error(std::string("Fabric_generic_grouped") + __func__ + ": Connection closed");
  }

  return ct_total;
}

std::size_t Fabric_generic_grouped::poll_completions_tentative(const Component::IFabric_op_completer::complete_tentative &cb_)
{
  std::size_t ct_total = 0;

  {
    std::lock_guard<std::mutex> k{_m_rxcq};
    ct_total += _rxcq.poll_completions_tentative(cb_);
  }
  {
    std::lock_guard<std::mutex> k{_m_txcq};
    ct_total += _txcq.poll_completions_tentative(cb_);
  }

  std::lock_guard<std::mutex> k{_m_cnxn};
  if ( _cnxn.is_shut_down() && ct_total == 0 )
  {
    throw std::logic_error(std::string("Fabric_generic_grouped") + __func__ + ": Connection closed");
  }

  return ct_total;
}

std::size_t Fabric_generic_grouped::poll_completions(const Component::IFabric_op_completer::complete_param_definite &cb_, void *cb_param_)
{
  std::size_t ct_total = 0;

  {
    std::lock_guard<std::mutex> k{_m_rxcq};
    ct_total += _rxcq.poll_completions(cb_, cb_param_);
  }
  {
    std::lock_guard<std::mutex> k{_m_txcq};
    ct_total += _txcq.poll_completions(cb_, cb_param_);
  }

  std::lock_guard<std::mutex> k{_m_cnxn};
  if ( _cnxn.is_shut_down() && ct_total == 0 )
  {
    throw std::logic_error(std::string("Fabric_generic_grouped") + __func__ + ": Connection closed");
  }

  return ct_total;
}

std::size_t Fabric_generic_grouped::poll_completions_tentative(const Component::IFabric_op_completer::complete_param_tentative &cb_, void *cb_param_)
{
  std::size_t ct_total = 0;

  {
    std::lock_guard<std::mutex> k{_m_rxcq};
    ct_total += _rxcq.poll_completions_tentative(cb_, cb_param_);
  }
  {
    std::lock_guard<std::mutex> k{_m_txcq};
    ct_total += _txcq.poll_completions_tentative(cb_, cb_param_);
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
  auto comm = new Fabric_comm_grouped(*this, this->_rxcq, this->_txcq);
  {
    std::lock_guard<std::mutex> g{_m_comms};
    _comms.insert(comm);
  }
  /* Each cq has a set of comms which it supports */
  {
    std::lock_guard<std::mutex> g{_m_rxcq};
    _rxcq.member_insert(&comm->rx());
  }
  {
    std::lock_guard<std::mutex> g{_m_txcq};
    _txcq.member_insert(&comm->tx());
  }
  return comm;
}

void Fabric_generic_grouped::forget_group(Fabric_comm_grouped *comm_)
{
  std::lock_guard<std::mutex> g{_m_comms};
  _comms.erase(comm_);
}

std::size_t Fabric_generic_grouped::max_message_size() const noexcept
{
  return _cnxn.max_message_size();
}

std::size_t Fabric_generic_grouped::max_inject_size() const noexcept
{
  return _cnxn.max_inject_size();
}
