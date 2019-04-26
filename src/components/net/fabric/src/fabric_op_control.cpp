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

#include "fabric_op_control.h"

#include "event_registration.h"
#include "fabric.h" /* trywait() */
#include "fabric_check.h" /* CHECK_FI_ERR */
#include "fabric_ptr.h" /* fid_unique_ptr */
#include "fabric_runtime_error.h"
#include "fabric_str.h" /* tostr */
#include "fabric_util.h" /* make_fi_infodup, get_event_name */
#include "fd_control.h"
#include "fd_pair.h"
#include "fd_unblock_set_monitor.h"
#include "system_fail.h"

#include <rdma/fi_errno.h> /* fi_strerror */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <rdma/fi_rma.h> /* fi_{read,recv,send,write}v, fi_inject */
#pragma GCC diagnostic pop

#include <sys/select.h> /* pselect */
#include <sys/uio.h> /* iovec */

#include <boost/io/ios_state.hpp>

#include <algorithm> /* min */
#include <chrono> /* milliseconds */
#include <cstdlib> /* getenv */
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
  , _cq_attr{100, 0U, Fabric_cq::fi_cq_format, FI_WAIT_SET, 0U, FI_CQ_COND_NONE, &*_wait_set}
#else
  , _cq_attr{100, 0U, Fabric_cq::fi_cq_format, FI_WAIT_FD, 0U, FI_CQ_COND_NONE, nullptr}
#endif
  , _rxcq(make_fid_cq(_cq_attr, this), "rx")
  , _txcq(make_fid_cq(_cq_attr, this), "tx")
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
  CHECK_FI_ERR(::fi_ep_bind(&*_ep, _txcq.fid(), FI_TRANSMIT));
  CHECK_FI_ERR(::fi_ep_bind(&*_ep, _rxcq.fid(), FI_RECV));
  CHECK_FI_ERR(::fi_enable(&*_ep));
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
  const ::iovec *first_
  , const ::iovec *last_
  , void **desc_
  , void *context_
)
{
  CHECK_FI_EQ(
    ::fi_sendv(
      &ep()
      , first_
      , desc_
      , std::size_t(last_ - first_)
      , ::fi_addr_t{}
      , context_
    )
    , 0
  );
  _txcq.incr_inflight(__func__);
}

void Fabric_op_control::post_send(
  const ::iovec *first_
  , const ::iovec *last_
  , void *context_
)
{
  auto desc = populated_desc(first_, last_);
  post_send(first_, last_, &*desc.begin(), context_);
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
  const ::iovec *first_
  , const ::iovec *last_
  , void **desc_
  , void *context_
)
{
  CHECK_FI_EQ(
    ::fi_recvv(
      &ep()
      , first_
      , desc_
      , std::size_t(last_ - first_)
      , ::fi_addr_t{}
      , context_
    )
    , 0
  );
  _rxcq.incr_inflight(__func__);
}
void Fabric_op_control::post_recv(
  const ::iovec *first_
  , const ::iovec *last_
  , void *context_
)
{
  auto desc = populated_desc(first_, last_);
  post_recv(first_, last_, &*desc.begin(), context_);
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
  const ::iovec *first_
  , const ::iovec *last_
  , void **desc_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  CHECK_FI_EQ(
    ::fi_readv(
      &ep()
      , first_
      , desc_
      , std::size_t(last_ - first_)
      , ::fi_addr_t{}
      , remote_addr_
      , key_
      , context_
    )
    , 0
  );
  _txcq.incr_inflight(__func__);
}

void Fabric_op_control::post_read(
  const ::iovec *first_
  , const ::iovec *last_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  auto desc = populated_desc(first_, last_);
  post_read(first_, last_, &*desc.begin(), remote_addr_, key_, context_);
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
  const ::iovec *first_
  , const ::iovec *last_
  , void **desc_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  CHECK_FI_EQ(
    ::fi_writev(
      &ep()
      , first_
      , desc_
      , std::size_t(last_ - first_)
      , ::fi_addr_t{}
      , remote_addr_
      , key_
      , context_
      )
    , 0
    );
  _txcq.incr_inflight(__func__);
}

void Fabric_op_control::post_write(
  const ::iovec *first_
  , const ::iovec *last_
  , uint64_t remote_addr_
  , uint64_t key_
  , void *context_
)
{
  auto desc = populated_desc(first_, last_);
  post_write(first_, last_, &*desc.begin(), remote_addr_, key_, context_);
}

  /**
   * Send message without completion
   *
   * @param connection Connection to inject on
   * @param buf_ start of data to send
   * @param len_ length of data to send (must not exceed max_inject_size())
   */
void Fabric_op_control::inject_send(const void *buf_, std::size_t len_)
{
  CHECK_FI_EQ(::fi_inject(&ep(), buf_, len_, ::fi_addr_t{}), 0);
}

/**
 * Poll completions (e.g., completions)
 *
 * @param completion_callback (context_t, ::status_t status, void* error_data)
 *
 * @return Number of completions processed
 */

#pragma GCC diagnostic push
#if defined __GNUC__ && 6 < __GNUC__
#pragma GCC diagnostic ignored "-Wnoexcept-type"
#endif

std::size_t Fabric_op_control::poll_completions(const Component::IFabric_op_completer::complete_old &cb_)
{
  std::size_t ct_total = 0;

  ct_total += _rxcq.poll_completions(cb_);
  ct_total += _txcq.poll_completions(cb_);

  if ( _shut_down && ct_total == 0 )
  {
    throw std::logic_error(__func__ + std::string(": Connection closed"));
  }
  return ct_total;
}

std::size_t Fabric_op_control::poll_completions(const Component::IFabric_op_completer::complete_definite &cb_)
{
  std::size_t ct_total = 0;

  ct_total += _rxcq.poll_completions(cb_);
  ct_total += _txcq.poll_completions(cb_);

  if ( _shut_down && ct_total == 0 )
  {
    throw std::logic_error(__func__ + std::string(": Connection closed"));
  }
  return ct_total;
}

std::size_t Fabric_op_control::poll_completions_tentative(const Component::IFabric_op_completer::complete_tentative &cb_)
{
  std::size_t ct_total = 0;

  ct_total += _rxcq.poll_completions_tentative(cb_);
  ct_total += _txcq.poll_completions_tentative(cb_);

  if ( _shut_down && ct_total == 0 )
  {
    throw std::logic_error(__func__ + std::string(": Connection closed"));
  }
  return ct_total;
}

std::size_t Fabric_op_control::poll_completions(const Component::IFabric_op_completer::complete_param_definite &cb_, void *cb_param_)
{
  std::size_t ct_total = 0;

  ct_total += _rxcq.poll_completions(cb_, cb_param_);
  ct_total += _txcq.poll_completions(cb_, cb_param_);

  if ( _shut_down && ct_total == 0 )
  {
    throw std::logic_error(__func__ + std::string(": Connection closed"));
  }
  return ct_total;
}

std::size_t Fabric_op_control::poll_completions_tentative(const Component::IFabric_op_completer::complete_param_tentative &cb_, void *cb_param_)
{
  std::size_t ct_total = 0;

  ct_total += _rxcq.poll_completions_tentative(cb_, cb_param_);
  ct_total += _txcq.poll_completions_tentative(cb_, cb_param_);

  if ( _shut_down && ct_total == 0 )
  {
    throw std::logic_error(__func__ + std::string(": Connection closed"));
  }
  return ct_total;
}

#pragma GCC diagnostic pop

/**
 * Block and wait for next completion.
 *
 * @param polls_limit Maximum number of polls (throws exception on exceeding limit)
 *
 * @return Next completion context
 * @throw std::system_error - creating fd pair
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
    static constexpr unsigned cq_count = 2;
    ::fid_t f[cq_count] = { _rxcq.fid(), _txcq.fid() };
    /* if fabric is in a state in which it can wait on the cqs ... */
    if ( fabric().trywait(f, cq_count) == FI_SUCCESS )
    {
      /* Wait sets: libfabric may notify any of read, write, except */
      fd_set fds_read;
      fd_set fds_write;
      fd_set fds_except;
      FD_ZERO(&fds_read);
      FD_ZERO(&fds_write);
      FD_ZERO(&fds_except);

      /* the fd through which an unblock call can end the wait */
      FD_SET(fd_unblock.fd_read(), &fds_read);

      auto fd_max = fd_unblock.fd_read(); /* max fd value */
      /* add the cq file descriptors to the wait set */
      for ( unsigned i = 0; i != cq_count; ++i )
      {
        int fd;
        CHECK_FI_ERR(::fi_control(f[i], FI_GETWAIT, &fd));
        FD_SET(fd, &fds_read);
        FD_SET(fd, &fds_write);
        FD_SET(fd, &fds_except);
        fd_max = std::max(fd_max, fd);
      }
      struct timespec ts {
        timeout.count() / 1000 /* seconds */
        , (timeout.count() % 1000) * 1000000 /* nanoseconds */
      };

      /* Wait until libfabric indicates a completion */
      auto ready = ::pselect(fd_max+1, &fds_read, &fds_write, &fds_except, &ts, nullptr);
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
       *  - the eventual completion will take care of them, or
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
    catch ( const fabric_runtime_error &e )
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
catch ( const fabric_runtime_error &e )
{
  throw e.add(tostr(info));
}

std::size_t Fabric_op_control::max_message_size() const noexcept
{
  return _ep_info->ep_attr->max_msg_size;
}

std::size_t Fabric_op_control::max_inject_size() const noexcept
{
  return _ep_info->tx_attr->inject_size;
}
