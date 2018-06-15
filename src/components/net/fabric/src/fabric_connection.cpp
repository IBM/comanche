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

#include "event_producer.h"
#include "event_registration.h"
#include "fabric.h"
#include "fabric_check.h"
#include "fabric_error.h"
#include "fabric_comm.h"
#include "fabric_ptr.h"
#include "fabric_util.h" /* make_fi_infodup, make_fid_cq, make_fid_aep, make_fid_mr_reg_ptr, get_event_name */
#include "fd_pair.h"
#include "fd_unblock_set_monitor.h"
#include "async_req_record.h"
#include "system_fail.h"

#include <rdma/fabric.h> /* FI_TYPE_xxx */
#include <rdma/fi_domain.h> /* fi_wait */
#include <rdma/fi_rma.h> /* fi_writev */

#include <sys/select.h> /* pselect */
#include <sys/uio.h> /* iovec */

#include <algorithm> /* min */
#include <cassert>
#include <chrono> /* seconds */
#include <iostream> /* cerr */
#include <limits> /* <int>::max */
#include <thread> /* sleep_for */

/**
 * Fabric/RDMA-based network component
 *
 */

#define CAN USE_WAIT_SETS 0

/* Note: the info is owned by the caller, and must be copied if it is to be saved. */
Fabric_connection::Fabric_connection(
    Fabric &fabric_
    , event_producer &ev_
    , fi_info &info_
    , std::unique_ptr<Fd_control> control_
    , fabric_types::addr_ep_t (*set_peer_early)(std::unique_ptr<Fd_control> control, fi_info &info)
  )
  : Fabric_comm(*this)
  , _fabric(fabric_)
  , _domain_info(make_fi_infodup(info_, "domain"))
  /* NOTE: "this" is returned for context when domain-level events appear in the event queue bound to the domain
   * and not bound to a more specific entity (an endpoint, mr, av, pr scalable_ep).
   */
  , _domain(_fabric.make_fid_domain(*_domain_info, this))
#if USE_WAIT_SETS
  /* verbs provider does not support wait sets */
  , _wait_attr{
    FI_WAIT_FD /* wait_obj type. verbs supports ony FI_WAIT_FD */
    , 0U /* flags, "must be set to 0 by the caller" */
  }
  , _wait_set(make_fid_wait(fabric_, _wait_attr))
#else
#if 0
  , _fds_read{}
  , _fds_write{}
  , _fds_except{}
#endif
#endif
  , _m_fd_unblock_set{}
  , _fd_unblock_set{}
#if USE_WAIT_SETS
  , _cq_attr{100, 0U, FI_CQ_FORMAT_CONTEXT, FI_WAIT_SET, 0U, FI_CQ_COND_NONE, &*_wait_set}
#else
  , _cq_attr{100, 0U, FI_CQ_FORMAT_CONTEXT, FI_WAIT_FD, 0U, FI_CQ_COND_NONE, nullptr}
#endif
  , _cq(make_fid_cq(*_domain, _cq_attr, this))
  , _ep_info(make_fi_infodup(*_domain_info, "endpoint construction"))
  , _peer_addr(set_peer_early(std::move(control_), *_ep_info))
  , _ep(make_fid_aep(*_domain, *_ep_info, this))
  , _m{}
  , _mr_addr_to_desc{}
  , _mr_desc_to_addr{}
  , _m_comms{}
  , _comms{}
  /* events */
  , _event_pipe{}
  /* NOTE: the various tests for type (FI_EP_MSG) should perhaps
   * move to derived classses.
   *                      connection  message boundaries  reliable
   * FI_EP_MSG:               Y               Y              Y
   * FI_EP_SOCK_STREAM:       Y               N              Y
   * FI_EP_RDM:               N               Y              Y
   * FI_EP_DGRAM:             N               Y              N
   * FI_EP_SOCK_DGRAM:        N               N              N
   */
  , _event_registration( ep_info().ep_attr->type == FI_EP_MSG ? new event_registration(ev_, *this, ep()) : nullptr )
  , _shut_down(false)
{
  CHECK_FI_ERR(fi_ep_bind(&*_ep, &_cq->fid, FI_TRANSMIT | FI_RECV));
  CHECK_FI_ERR(fi_enable(&*_ep));
}

Fabric_connection::~Fabric_connection()
{
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
  /* recover the memory region as a unique ptr */
  auto mr = fid_ptr(static_cast<::fid_mr *>(std::get<0>(mr_)));

  {
    auto desc = ::fi_mr_desc(&*mr);
    guard g{_m};
    auto itr = _mr_desc_to_addr.find(desc);
    assert(itr != _mr_desc_to_addr.end());
    _mr_addr_to_desc.erase(itr->first);
    _mr_desc_to_addr.erase(itr);
  }
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
  CHECK_FI_ERR(fi_sendv(&*_ep, &*buffers.begin(), &*desc.begin(), buffers.size(), fi_addr_t{}, context));
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
  CHECK_FI_ERR(fi_recvv(&*_ep, &*buffers.begin(), &*desc.begin(), buffers.size(), fi_addr_t{}, context));
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
  CHECK_FI_ERR(
    fi_readv(
      &*_ep
      , &*buffers.begin(), &*desc.begin()
      , buffers.size(), fi_addr_t{}, remote_addr, key, context
    )
  );
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
  CHECK_FI_ERR(::fi_writev(&*_ep, &*buffers.begin(), &*desc.begin(),
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
  CHECK_FI_ERR(::fi_inject(&*_ep, &*buffers.begin(), buffers.size(), fi_addr_t{}));
}

void *Fabric_connection::get_cq_comp_err() const
{
  fi_cq_err_entry err{0,0,0,0,0,0,0,0,0,0,0};
  CHECK_FI_ERR(cq_readerr(&err, 0));

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
  return err.op_context;
}

std::size_t Fabric_connection::process_cq_comp_err(std::function<void(void *connection, status_t st)> completion_callback)
{
  completion_callback(get_cq_comp_err(), E_FAIL);
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
#if 1
    auto timeout = 0; /* immediate timeout */
    auto ct = cq_sread(&entry, ct_max, nullptr, timeout);
#else
    auto ct = cq_read(&entry, ct_max);
#endif
    if ( ct < 0 )
    {
      switch ( auto e = unsigned(-ct) )
      {
      case FI_EAVAIL:
        ct_total += process_cq_comp_err(cb_);
        break;
      case FI_EAGAIN:
        drained = true;
        break;
      default:
        throw fabric_error(e, __FILE__, __LINE__);
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

  if ( _shut_down && ct_total == 0 )
  {
    throw std::logic_error("Connection closed");
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
  Fd_pair fd_unblock;
  fd_unblock_set_monitor(_m_fd_unblock_set, _fd_unblock_set, fd_unblock.fd_write());
  /* Only block if we have not seen FI_SHUTDOWN */
  if ( ! _shut_down )
  {
/* verbs provider does not support wait sets */
#if USE_WAIT_SETS
    fi_wait(&*_wait_set, std::min(std::numeric_limits<int>::max(), int(timeout.count())));
#else
    ::fid_t f[1] = { &_cq->fid };
    if ( _fabric.trywait(f, 1) == FI_SUCCESS )
    {
      int fd;
      CHECK_FI_ERR(fi_control(&_cq->fid, FI_GETWAIT, &fd));
      fd_set fds_read;
      fd_set fds_write;
      fd_set fds_except;
      FD_ZERO(&fds_read);
      FD_SET(fd, &fds_read);
      FD_SET(fd_unblock.fd_read(), &fds_read);
      FD_ZERO(&fds_write);
      FD_SET(fd, &fds_write);
      FD_ZERO(&fds_except);
      FD_SET(fd, &fds_except);
      struct timespec ts {
        timeout.count() / 1000 /* seconds */
        , (timeout.count() % 1000) * 1000000 /* nanoseconds */
      };

      auto ready = ::pselect(std::max(fd,fd_unblock.fd_read())+1, &fds_read, &fds_write, &fds_except, &ts, nullptr);
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
  }
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

void Fabric_connection::forget_group(Fabric_comm *comm)
{
  std::lock_guard<std::mutex> g{_m_comms};
  _comms.erase(comm);
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
    auto sz = ::write(fd, &c, 1);
    (void) sz;
  }
}

auto Fabric_connection::get_name() const -> fabric_types::addr_ep_t
{
  auto it = static_cast<const char *>(_ep_info->src_addr);
  return fabric_types::addr_ep_t(
     std::vector<char>(it, it+_ep_info->src_addrlen)
  );
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

void Fabric_connection::queue_completion(Fabric_comm *comm_, void *context_, status_t status_)
{
  std::lock_guard<std::mutex> k{_m_comms};
  auto it = _comms.find(comm_);
  assert(it != _comms.end());
  (*it)->queue_completion(context_, status_);
}

#if 1
ssize_t Fabric_connection::cq_sread(void *buf, size_t count, const void *cond, int timeout) noexcept
{
  return ::fi_cq_sread(&*_cq, buf, count, cond, timeout);
}
#else
ssize_t Fabric_connection::cq_read(void *buf, size_t count) noexcept
{
  return ::fi_cq_read(&*_cq, buf, count);
}
#endif
ssize_t Fabric_connection::cq_readerr(fi_cq_err_entry *buf, uint64_t flags) const noexcept
{
  return ::fi_cq_readerr(&*_cq, buf, flags);
}

/* EVENTS */
void Fabric_connection::cb(std::uint32_t event, fi_eq_cm_entry &) noexcept
try
{
  if ( event == FI_SHUTDOWN )
  {
    _shut_down = true;
    unblock_completions();
  }
  _event_pipe.write(&event, sizeof event);
}
catch ( const std::exception &e )
{
  std::cerr << __func__ << " (Fabric_connection) " << e.what() << "\n";
}

void Fabric_connection::err(fi_eq_err_entry &) noexcept
try
{
  /* An error event; not what we were expecting */
  std::uint32_t event = FI_NOTIFY;
  _event_pipe.write(&event, sizeof event);
}
catch ( const std::exception &e )
{
  std::cerr << __func__ << " (Fabric_connection) " << e.what() << "\n";
}

void Fabric_connection::ensure_event() const
{
  /* First, ensure that expect_event will see an event */
  for ( bool have_event = false
    ; ! have_event
    ;
      /* Make some effort to wait until the event queue is readable.
       * NOTE: Seems to block for too long, as fi_trywait is not
       * behaving as expected. See Fabric::wait_eq.
       */
      wait_event()
  )
  {
    solicit_event(); /* _ev.read_eq() in client, no-op in server */
    auto fd = _event_pipe.fd_read();
    fd_set fds_read;
    FD_ZERO(&fds_read);
    FD_SET(fd, &fds_read);
    struct timespec ts {
      0 /* seconds */
      , 0 /* nanoseconds */
    };

    auto ready = ::pselect(fd+1, &fds_read, nullptr, nullptr, &ts, nullptr);

    if ( -1 == ready )
    {
      switch ( auto e = errno )
      {
      case EINTR:
        break;
      default:
        system_fail(e, "expect_event_sync");
      }
    }
    /* Note: there is no reason to act on the fd because
     *  - expect_event will read it
     */
    else
    {
      have_event = FD_ISSET(fd, &fds_read);
#if 0
      if ( have_event )
      {
        std::cerr << __func__ << " ready count " << ready << "\n";
      }
      else
      {
        std::cerr << __func__ << " timeout, ready count " << ready << "\n";
      }
#endif
    }
  }
}

void Fabric_connection::expect_event(std::uint32_t event_exp) const
{
  std::uint32_t event = 0;
  _event_pipe.read(&event, sizeof event);
  if ( event != event_exp )
  {
    throw std::logic_error(std::string("expected ") + get_event_name(event_exp) + " got " + get_event_name(event) );
  }
}
