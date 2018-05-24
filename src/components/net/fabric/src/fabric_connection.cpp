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

#include "fabric_connection.h"

#include "fabric_error.h"
#include "fabric_comm.h"
#include "fabric_util.h"
#include "fd_unblock_set_monitor.h"
#include "async_req_record.h"
#include "system_fail.h"

#include <rdma/fabric.h> /* FI_TYPE_xxx */
#include <rdma/fi_cm.h> /* fi_connect */
#include <rdma/fi_domain.h> /* fi_wait */
#include <rdma/fi_rma.h> /* fi_writev */

#include <sys/select.h> /* pselect */
#include <sys/uio.h> /* iovec */

#include <algorithm> /* min */
#include <limits> /* <int>::max */

/**
 * Fabric/RDMA-based network component
 *
 */

/* Note: the info is owned by the caller, and must be copied if it is to be saved. */
Fabric_connection::Fabric_connection(fid_fabric &fabric_, fid_eq &eq_, fi_info &info_, Fd_control &&control_, fabric_types::addr_ep_t (*set_peer_early)(Fd_control &control, fi_info &info))
  : Fabric_comm(*this)
  , _control(std::move(control_))
  , _domain_info(make_fi_infodup(info_, "domain"))
  /* NOTE: "this" is returned for context when domain-level events appear in the event queue bound to the domain
   * and not bound to a more specific entity (an endpoint, mr, av, pr scalable_ep).
   */
  , _domain(make_fid_domain(fabric_, *_domain_info, this))
#if 0
  /* verbs provider does not support wait sets */
  , _wait_attr{
    FI_WAIT_FD /* wait_obj type. verbs supports ony FI_WAIT_FD */
    , 0U /* flags, "must be set to 0 by the caller" */
  }
  , _wait_set(make_fid_wait(fabric_, _wait_attr))
  , _cq_attr{100, 0U, FI_CQ_FORMAT_CONTEXT, FI_WAIT_SET, 0U, FI_CQ_COND_NONE, &*_wait_set}
#else
  , _cq_attr{100, 0U, FI_CQ_FORMAT_CONTEXT, FI_WAIT_FD, 0U, FI_CQ_COND_NONE, nullptr}
#endif
  , _cq(make_fid_cq(*_domain, _cq_attr, this))
  , _ep_info(make_fi_infodup(*_domain_info, "endpoint construction"))
  , _peer_addr(set_peer_early(_control, *_ep_info))
  , _ep(make_fid_aep(*_domain, *_ep_info, this))
{
  /* NOTE: the various tests for type (FI_EP_MSG) should perhaps
   * move to derived classses.
   *                      connection  message boundaries  reliable
   * FI_EP_MSG:               Y               Y              Y
   * FI_EP_SOCK_STREAM:       Y               N              Y
   * FI_EP_RDM:               N               Y              Y
   * FI_EP_DGRAM:             N               Y              N
   * FI_EP_SOCK_DGRAM:        N               N              N
   */
  if ( _ep_info->ep_attr->type == FI_EP_MSG )
  {
    CHECKZ(fi_ep_bind(&*_ep, &eq_.fid, 0));
  }

  CHECKZ(fi_ep_bind(&*_ep, &_cq->fid, FI_TRANSMIT | FI_RECV));

  CHECKZ(fi_enable(&*_ep));
}

Fabric_connection::~Fabric_connection()
{
  /* "the flags parameter is reserved and must be 0" */
  fi_shutdown(&*_ep, 0);
}

/**
 * Register buffer for RDMA
 *
 * @param contig_addr Pointer to contiguous region
 * @param size Size of buffer in bytes
 * @param flags Flags e.g., FI_REMOTE_READ|FI_REMOTE_WRITE
 *
 * @return Memory region handle
 */
auto Fabric_connection::register_memory(const void * contig_addr, size_t size, std::uint64_t key, std::uint64_t flags) -> Component::IFabric_connection::memory_region_t
{
  auto mr = make_fid_mr_reg_ptr(*_domain, contig_addr, size, std::uint64_t(FI_SEND|FI_RECV|FI_READ|FI_WRITE|FI_REMOTE_READ|FI_REMOTE_WRITE), key, flags);

  /* operations which access local memory will need the memory "descriptor." Record it here. */
  auto desc = ::fi_mr_desc(mr);
  {
    guard g{_m};
    _mr_addr_to_desc[contig_addr] = desc;
    _mr_desc_to_addr[desc] = contig_addr;
  }

  /*
   * Operations which access remote memory will need the memory key.
   * If the domain has FI_MR_PROV_KEY set, we need to return the actual key.
   */
  return Component::IFabric_connection::memory_region_t{mr, ::fi_mr_key(mr)};
}

  /**
   * De-register memory region
   *
   * @param memory_region
   */
void Fabric_connection::deregister_memory(const memory_region_t mr_)
{
  auto mr = static_cast<fid_mr *>(std::get<0>(mr_));

  {
    auto desc = ::fi_mr_desc(mr);
    guard g{_m};
    auto itr = _mr_desc_to_addr.find(desc);
    assert(itr != _mr_desc_to_addr.end());
    _mr_addr_to_desc.erase(itr->first);
    _mr_desc_to_addr.erase(itr);
  }

  fi_close(&mr->fid);
}

/* If local keys are needed, one local key per buffer. */
std::vector<void *> Fabric_connection::populated_desc(const std::vector<iovec> & buffers)
{
  std::vector<void *> desc;
  for ( const auto it : buffers )
  {
    {
      guard g{_m};
      /* find a key equal to k or, if none, the largest key less than k */
      auto dit = _mr_addr_to_desc.lower_bound(it.iov_base);
      /* If not at k, lower_bound has left us with an iterator beyond k. Back up */
      if ( dit->first != it.iov_base && dit != _mr_addr_to_desc.begin() )
      {
        --dit;
      }

      desc.emplace_back(dit->second);
    }
  }
  return desc;
}

/**
 * Asynchronously post a buffer to the connection
 *
 * @param connection Connection to send on
 * @param buffers Buffer vector (containing regions should be registered)
 *
 * @return Work (context) identifier
 */
void Fabric_connection::post_send_internal(
  const std::vector<iovec>& buffers, void *context)
{
  auto desc = populated_desc(buffers);
  CHECKZ(fi_sendv(&*_ep, &*buffers.begin(), &*desc.begin(), buffers.size(), fi_addr_t{}, context));
}

/**
 * Asynchronously post a buffer to receive data
 *
 * @param connection Connection to post to
 * @param buffers Buffer vector (containing regions should be registered)
 *
 * @return Work (context) identifier
 */
void Fabric_connection::post_recv_internal(
  const std::vector<iovec>& buffers, void *context)
{
  auto desc = populated_desc(buffers);
  CHECKZ(fi_recvv(&*_ep, &*buffers.begin(), &*desc.begin(), buffers.size(), fi_addr_t{}, context));
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
void Fabric_connection::post_read_internal(
  const std::vector<iovec>& buffers,
  uint64_t remote_addr,
  uint64_t key,
  void *context)
{
  auto desc = populated_desc(buffers);
  CHECKZ(fi_readv(&*_ep, &*buffers.begin(), &*desc.begin(),
    buffers.size(), fi_addr_t{}, remote_addr, key, context));
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
void Fabric_connection::post_write_internal(
  const std::vector<iovec>& buffers,
  uint64_t remote_addr,
  uint64_t key,
  void *context)
{
  auto desc = populated_desc(buffers);
  CHECKZ(::fi_writev(&*_ep, &*buffers.begin(), &*desc.begin(),
    buffers.size(), fi_addr_t{}, remote_addr, key, context));
}

  /**
   * Send message without completion
   *
   * @param connection Connection to inject on
   * @param buffers Buffer vector (containing regions should be registered)
   */
void Fabric_connection::inject_send(const std::vector<iovec>& buffers)
{
  CHECKZ(::fi_inject(&*_ep, &*buffers.begin(), buffers.size(), fi_addr_t{}));
}

#include <iostream>
std::size_t get_cq_comp_err(fid_cq &cq_, std::function<void(void *connection, status_t st)> completion_callback)
{
  fi_cq_err_entry err{0,0,0,0,0,0,0,0,0,0,0};
  CHECKZ(fi_cq_readerr(&cq_, &err, 0));

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
  completion_callback(err.op_context, E_FAIL);

  return 1U;
}

/**
 * Poll completions (e.g., completions)
 *
 * @param completion_callback (context_t, status_t status, void* error_data)
 *
 * @return Number of completions processed
 */

std::size_t Fabric_connection::poll_completions(std::function<void(void *context, status_t st) noexcept> cb_)
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
    switch ( auto ct = cq_sread(&entry, ct_max, nullptr, 0) )
    {
    case -FI_EAVAIL:
      ct_total += ::get_cq_comp_err(*_cq, cb_);
      break;
    case -FI_EAGAIN:
      drained = true;
      break;
    default:
      if ( ct < 0 )
      {
        throw fabric_error(int(-ct), __FILE__, __LINE__);
      }

      std::unique_ptr<async_req_record> g_context(static_cast<async_req_record *>(entry.op_context));
      cb_(g_context->context(), S_OK);
      ++ct_total;

      g_context.release();
      break;
    }
  }

  return ct_total;
}

std::size_t Fabric_connection::stalled_completion_count()
{
  return 0U; /* completions which are not part of an allocated comm do not stall */
}

/**
 * Block and wait for next completion.
 *
 * @param polls_limit Maximum number of polls (throws exception on exceeding limit)
 *
 * @return Next completion context
 */
void Fabric_connection::wait_for_next_completion(std::chrono::milliseconds timeout)
{
/* verbs provider does not support wait sets */
#if 0
  fi_wait(&*_wait_set, std::min(std::numeric_limits<int>::max(), int(timeout.count())));
#else
  int fd;
  CHECKZ(fi_control(&_cq->fid, FI_GETWAIT, &fd));
  FD_ZERO(&_fds_read);
  FD_SET(fd, &_fds_read);
  Fd_pair fd_unblock;
  fd_unblock_set_monitor(_m_fd_unblock_set, _fd_unblock_set, fd_unblock.fd_write());
  FD_SET(fd_unblock.fd_read(), &_fds_read);
  FD_ZERO(&_fds_write);
  FD_SET(fd, &_fds_write);
  FD_ZERO(&_fds_except);
  FD_SET(fd, &_fds_except);
  struct timespec ts {
    timeout.count() / 1000 /* seconds */
    , (timeout.count() % 1000) * 1000000 /* nanoseconds */
  };

  auto ready = ::pselect(fd+1, &_fds_read, &_fds_write, &_fds_except, &ts, nullptr);
  if ( -1 == ready )
  {
    switch ( auto e = errno )
    {
    case EINTR:
      break;
    default:
      system_fail(e, "wait_for_next_completion");
    }
  }
  /* Note: there is no reason to act on the fd's because either
   *  - fi_cq_read will take care of them, or
   *  - the fd_unblock_set_monitor will take care of them.
   */
#endif
}

void Fabric_connection::wait_for_next_completion(unsigned polls_limit)
{
  for ( ; polls_limit != 0; --polls_limit )
  {
    try
    {
      return wait_for_next_completion(std::chrono::milliseconds(0));
    }
    catch ( const fabric_error &e )
    {
      if ( e.id() != FI_ETIMEDOUT )
      {
        throw;
      }
    }
  }
}

auto Fabric_connection::allocate_group() -> IFabric_communicator *
{
  std::lock_guard<std::mutex> g{_m_comms};
  auto comm = new Fabric_comm(*this);
  _comms.insert(comm);
  return comm;
}

/**
 * Unblock any threads waiting on completions
 *
 */
void Fabric_connection::unblock_completions()
{
  std::lock_guard<std::mutex> g{_m_fd_unblock_set};
  for ( auto fd : _fd_unblock_set )
  {
    char c{};
    ::write(fd, &c, 1);
  }
}

  /**
   * Get address of connected peer (taken from fi_getpeer during
   * connection instantiation).
   *
   *
   * @return Peer endpoint address
   */
std::string Fabric_connection::get_peer_addr()
{
  return std::string(_peer_addr.begin(), _peer_addr.end());
}

  /**
   * Get local address of connection (taken from fi_getname during
   * connection instantiation).
   *
   *
   * @return Local endpoint address
   */

std::string Fabric_connection::get_local_addr()
{
  auto v = get_name();
  return std::string(v.begin(), v.end());
}

auto Fabric_connection::get_name() const -> fabric_types::addr_ep_t
{
  auto it = static_cast<const char *>(_ep_info->src_addr);
  return fabric_types::addr_ep_t(
     std::vector<char>(it, it+_ep_info->src_addrlen)
  );
}

void Fabric_connection::await_connected(fid_eq &eq_) const
{
  fi_eq_cm_entry entry;
  std:: uint32_t event;
  CHECKZ(fi_eq_sread(&eq_, &event, &entry, sizeof entry, -1, 0));
  assert(event == FI_CONNECTED);
}

std::size_t Fabric_connection::get_cq_comp_err(std::function<void(void *connection, status_t st)> completion_callback)
{
  return ::get_cq_comp_err(*_cq, completion_callback);
}

ssize_t Fabric_connection::cq_sread(void *buf, size_t count, const void *cond, int timeout) noexcept
{
  return fi_cq_sread(&*_cq, buf, count, cond, timeout);
}

ssize_t Fabric_connection::cq_readerr(fi_cq_err_entry *buf, uint64_t flags) noexcept
{
  return fi_cq_readerr(&*_cq, buf, flags);
}

void Fabric_connection::queue_completion(Fabric_comm *comm_, void *context_, status_t status_)
{
  std::lock_guard<std::mutex> k{_m_comms};
  auto it = _comms.find(comm_);
  assert(it != _comms.end());
  (*it)->queue_completion(context_, status_);
}
