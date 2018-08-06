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

#include "fabric_op_control.h"

#include "event_registration.h"
#include "fabric.h" /* trywait() */
#include "fabric_check.h" /* CHECK_FI_ERR */
#include "fabric_error.h"
#include "fabric_ptr.h" /* fid_unique_ptr */
#include "fabric_str.h" /* tostr */
#include "fabric_util.h" /* make_fi_infodup, get_event_name */
#include "fd_control.h"
#include "fd_pair.h"
#include "fd_unblock_set_monitor.h"
#include "system_fail.h"

#include <rdma/fi_errno.h> /* fi_strerror */
#include <rdma/fi_rma.h> /* fi_{read,recv,send,write}v, fi_inject */

#include <sys/select.h> /* pselect */
#include <sys/uio.h> /* iovec */

#include <boost/io/ios_state.hpp>

#include <algorithm> /* min */
#include <chrono> /* milliseconds */
#include <iostream> /* cerr */
#include <limits> /* <int>::max */

/**
 * Fabric/RDMA-based network component
 *
 */

#define CAN_USE_WAIT_SETS 0

/* Note: the info is owned by the caller, and must be copied if it is to be saved. */
Fabric_op_control::Fabric_op_control(
    Fabric &fabric_
    , event_producer &ev_
    , ::fi_info &info_
    , std::unique_ptr<Fd_control> control_
    , fabric_types::addr_ep_t (*set_peer_early_)(std::unique_ptr<Fd_control> control, ::fi_info &info)
  )
  : Fabric_memory_control(
      fabric_, info_
  )
  , _m_completions{}
  , _completions{}
#if CAN_USE_WAIT_SETS
  /* verbs provider does not support wait sets */
  , _wait_attr{
    FI_WAIT_FD /* wait_obj type. verbs supports ony FI_WAIT_FD */
    , 0U /* flags, "must be set to 0 by the caller" */
  }
  , _wait_set(make_fid_wait(fabric(), _wait_attr))
#endif
  , _m_fd_unblock_set{}
  , _fd_unblock_set{}
#if CAN_USE_WAIT_SETS
  , _cq_attr{100, 0U, FI_CQ_FORMAT_CONTEXT, FI_WAIT_SET, 0U, FI_CQ_COND_NONE, &*_wait_set}
#else
  , _cq_attr{100, 0U, FI_CQ_FORMAT_CONTEXT, FI_WAIT_FD, 0U, FI_CQ_COND_NONE, nullptr}
#endif
  , _cq(make_fid_cq(_cq_attr, this))
  , _ep_info(make_fi_infodup(domain_info(), "endpoint construction"))
  , _peer_addr(set_peer_early_(std::move(control_), *_ep_info))
  , _ep(make_fid_aep(*_ep_info, this))
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
/* ERROR (probably in libfabric verbs): closing an active endpoint prior to fi_ep_bind causes a SEGV,
 * as fi_ibv_msg_ep_close will call fi_ibv_cleanup_cq whether there is CQ state to clean up.
 */
  CHECK_FI_ERR(fi_ep_bind(&*_ep, &_cq->fid, FI_TRANSMIT | FI_RECV));
  CHECK_FI_ERR(fi_enable(&*_ep));
}

Fabric_op_control::~Fabric_op_control()
{
}

/**
 * Asynchronously post a buffer to the connection
 *
 * @param connection Connection to send on
 * @param buffers Buffer vector (containing regions should be registered)
 *
 * @return Work (context) identifier
 */
void Fabric_op_control::post_send(
  const std::vector<iovec>& buffers_
  , void *context_
)
{
  auto desc = populated_desc(buffers_);
  CHECK_FI_ERR(
    ::fi_sendv(
      &ep()
      , &*buffers_.begin()
      , &*desc.begin()
      , buffers_.size()
      , ::fi_addr_t{}
      , context_
    )
  );
}

/**
 * Asynchronously post a buffer to receive data
 *
 * @param connection Connection to post to
 * @param buffers Buffer vector (containing regions should be registered)
 *
 * @return Work (context) identifier
 */
void Fabric_op_control::post_recv(
  const std::vector<iovec>& buffers_
  , void *context_
)
{
  auto desc = populated_desc(buffers_);
  CHECK_FI_ERR(
    ::fi_recvv(
      &ep()
      , &*buffers_.begin()
      , &*desc.begin()
      , buffers_.size()
      , ::fi_addr_t{}
      , context_
    )
  );
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
void Fabric_op_control::post_read(
  const std::vector<iovec>& buffers_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  auto desc = populated_desc(buffers_);
  CHECK_FI_ERR(
    ::fi_readv(
      &ep()
      , &*buffers_.begin()
      , &*desc.begin()
      , buffers_.size()
      , ::fi_addr_t{}
      , remote_addr_
      , key_
      , context_
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
void Fabric_op_control::post_write(
  const std::vector<iovec>& buffers_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  auto desc = populated_desc(buffers_);
  CHECK_FI_ERR(
    ::fi_writev(
      &ep()
      , &*buffers_.begin()
      , &*desc.begin()
      , buffers_.size()
      , ::fi_addr_t{}
      , remote_addr_
      , key_
      , context_
      )
    );
}

  /**
   * Send message without completion
   *
   * @param connection Connection to inject on
   * @param buffers Buffer vector (containing regions should be registered)
   */
void Fabric_op_control::inject_send(const std::vector<iovec>& buffers)
{
  CHECK_FI_ERR(::fi_inject(&ep(), &*buffers.begin(), buffers.size(), ::fi_addr_t{}));
}

void *Fabric_op_control::get_cq_comp_err() const
{
  ::fi_cq_err_entry err{0,0,0,0,0,0,0,0,0,0,0};
  CHECK_FI_ERR(cq_readerr(&err, 0));

  boost::io::ios_base_all_saver sv(std::cerr);
  std::cerr << __func__ << " : "
                  << " op_context " << err.op_context
                  << std::hex
                  << " flags " << err.flags
                  << std::dec
                  << " len " << err.len
                  << " buf " << err.buf
                  << " data " << err.data
                  << " tag " << err.tag
                  << " olen " << err.olen
                  << " err " << err.err
                  << " (text) " << ::fi_strerror(err.err)
                  << " errno " << err.prov_errno
                  << " err_data " << err.err_data
                  << " err_data_size " << err.err_data_size
        << std::endl;
  return err.op_context;
}

std::size_t Fabric_op_control::process_or_queue_completion(void *context_, std::function<cb_acceptance(void *context, status_t st)> cb_, status_t status_)
{
  std::size_t ct_total = 0U;
  if ( cb_(context_, status_) == cb_acceptance::ACCEPT )
  {
    ++ct_total;
  }
  else
  {
    queue_completion(context_, status_);
  }

  return ct_total;
}

std::size_t Fabric_op_control::process_cq_comp_err(std::function<void(void *context, status_t st)> cb_)
{
  cb_(get_cq_comp_err(), E_FAIL);
  return 1U;
}

std::size_t Fabric_op_control::process_or_queue_cq_comp_err(std::function<cb_acceptance(void *context, status_t st)> cb_)
{
  return process_or_queue_completion(get_cq_comp_err(), cb_, E_FAIL);
}

/**
 * Poll completions (e.g., completions)
 *
 * @param completion_callback (context_t, status_t status, void* error_data)
 *
 * @return Number of completions processed
 */

std::size_t Fabric_op_control::poll_completions(std::function<void(void *context, status_t st) noexcept> cb_)
{
  std::size_t constexpr ct_max = 1;
  std::size_t ct_total = 0;
  ::fi_cq_tagged_entry entry; /* We dont actually expect a tagged entry. Spefifying this to provide the largest buffer. */
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
      cb_(entry.op_context, S_OK);
      ++ct_total;
    }
  }

  ct_total += drain_old_completions(cb_);

  if ( _shut_down && ct_total == 0 )
  {
    throw std::logic_error("Connection closed");
  }
  return ct_total;
}

std::size_t Fabric_op_control::poll_completions_tentative(std::function<cb_acceptance(void *context, status_t st) noexcept> cb_)
{
  std::size_t constexpr ct_max = 1;
  std::size_t ct_total = 0;
  ::fi_cq_tagged_entry entry; /* We dont actually expect a tagged entry. Specifying this to provide the largest buffer. */
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
      ct_total += process_or_queue_completion(entry.op_context, cb_, S_OK);
    }
  }

  ct_total += drain_old_completions(cb_);

  if ( _shut_down && ct_total == 0 )
  {
    throw std::logic_error("Connection closed");
  }
  return ct_total;
}

/**
 * Block and wait for next completion.
 *
 * @param polls_limit Maximum number of polls (throws exception on exceeding limit)
 *
 * @return Next completion context
 */
void Fabric_op_control::wait_for_next_completion(std::chrono::milliseconds timeout)
{
  Fd_pair fd_unblock;
  fd_unblock_set_monitor(_m_fd_unblock_set, _fd_unblock_set, fd_unblock.fd_write());
  /* Only block if we have not seen FI_SHUTDOWN */
  if ( ! _shut_down )
  {
/* verbs provider does not support wait sets */
#if USE_WAIT_SETS
    ::fi_wait(&*_wait_set, std::min(std::numeric_limits<int>::max(), int(timeout.count())));
#else
    ::fid_t f[1] = { &_cq->fid };
    if ( fabric().trywait(f, 1) == FI_SUCCESS )
    {
      int fd;
      CHECK_FI_ERR(::fi_control(&_cq->fid, FI_GETWAIT, &fd));
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

void Fabric_op_control::wait_for_next_completion(unsigned polls_limit)
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

/**
 * Unblock any threads waiting on completions
 *
 */
void Fabric_op_control::unblock_completions()
{
  std::lock_guard<std::mutex> g{_m_fd_unblock_set};
  for ( auto fd : _fd_unblock_set )
  {
    char c{};
    auto sz = ::write(fd, &c, 1);
    (void) sz;
  }
}

auto Fabric_op_control::get_name() const -> fabric_types::addr_ep_t
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
std::string Fabric_op_control::get_peer_addr()
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

std::string Fabric_op_control::get_local_addr()
{
  auto v = get_name();
  return std::string(v.begin(), v.end());
}

ssize_t Fabric_op_control::cq_sread(void *buf, size_t count, const void *cond, int timeout) noexcept
{
  return ::fi_cq_sread(&*_cq, buf, count, cond, timeout);
}

ssize_t Fabric_op_control::cq_readerr(::fi_cq_err_entry *buf, uint64_t flags) const noexcept
{
  return ::fi_cq_readerr(&*_cq, buf, flags);
}

/* EVENTS */
void Fabric_op_control::cb(std::uint32_t event, ::fi_eq_cm_entry &) noexcept
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
  std::cerr << __func__ << " (Fabric_op_control) " << e.what() << "\n";
}

void Fabric_op_control::err(::fi_eq_err_entry &e_) noexcept
try
{
  /* An error event; not what we were expecting */
  std::cerr << "Fabric error event "
    << " fid " << e_.fid
    << "context " << e_.context
    << " data " << e_.data
    << " err " << e_.err
    << " prov_errno " << e_.prov_errno
    << " err_data ";
    boost::io::ios_base_all_saver sv(std::cerr);
    for ( auto i = static_cast<uint8_t *>(e_.err_data)
      ; i != static_cast<uint8_t *>(e_.err_data) + e_.err_data_size
      ; ++i
    )
    {
      std::cerr << std::setfill('0') << std::setw(2) << std::hex << *i;
    }
    std::cerr << std::endl;

  std::uint32_t event = FI_NOTIFY;
  _event_pipe.write(&event, sizeof event);
}
catch ( const std::exception &e )
{
  std::cerr << __func__ << " (Fabric_op_control) " << e.what() << "\n";
}

void Fabric_op_control::ensure_event() const
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

void Fabric_op_control::expect_event(std::uint32_t event_exp) const
{
  std::uint32_t event = 0;
  _event_pipe.read(&event, sizeof event);
  if ( event != event_exp )
  {
    throw std::logic_error(std::string("expected ") + get_event_name(event_exp) + " got " + get_event_name(event) );
  }
}

fid_unique_ptr<::fid_cq> Fabric_op_control::make_fid_cq(::fi_cq_attr &attr, void *context) const
{
  ::fid_cq *f;
  CHECK_FI_ERR(::fi_cq_open(&domain(), &attr, &f, context));
  FABRIC_TRACE_FID(f);
  return fid_unique_ptr<::fid_cq>(f);
}

std::shared_ptr<::fid_ep> Fabric_op_control::make_fid_aep(::fi_info &info, void *context) const
try
{
  ::fid_ep *f;
  CHECK_FI_ERR(::fi_endpoint(&domain(), &info, &f, context));
  static_assert(0 == FI_SUCCESS, "FI_SUCCESS not 0, which means that we need to distinguish between these types of \"successful\" returns");
  FABRIC_TRACE_FID(f);
  return fid_ptr(f);
}
catch ( const fabric_error &e )
{
  throw e.add(tostr(info));
}

void Fabric_op_control::queue_completion(void *context_, status_t status_)
{
  std::lock_guard<std::mutex> k2{_m_completions};
  _completions.push(completion_t(context_, status_));
}

std::size_t Fabric_op_control::drain_old_completions(std::function<void(void *context, status_t st) noexcept> completion_callback)
{
  std::size_t ct_total = 0U;
  std::unique_lock<std::mutex> k{_m_completions};
  while ( ! _completions.empty() )
  {
    auto c = _completions.front();
    _completions.pop();
    k.unlock();
    completion_callback(std::get<0>(c), std::get<1>(c));
    ++ct_total;
    k.lock();
  }
  return ct_total;
}

std::size_t Fabric_op_control::drain_old_completions(std::function<cb_acceptance(void *context, status_t st) noexcept> completion_callback)
{
  std::size_t ct_total = 0U;
  std::unique_lock<std::mutex> k{_m_completions};
  std::queue<completion_t> deferred_completions;
  while ( ! _completions.empty() )
  {
    auto c = _completions.front();
    _completions.pop();
    k.unlock();
    if ( completion_callback(std::get<0>(c), std::get<1>(c)) == cb_acceptance::ACCEPT )
    {
      ++ct_total;
    }
    else
    {
      deferred_completions.push(c);
    }
    k.lock();
  }
  std::swap(deferred_completions, _completions);
  return ct_total;
}
