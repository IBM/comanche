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

#include "fabric_connection_server_factory.h"

#include "event_producer.h"
#include "fabric.h"
#include "fabric_check.h"
#include "fabric_connection.h"
#include "fabric_connection_server.h"
#include "fabric_util.h" /* get_name */
#include "pointer_cast.h"
#include "system_fail.h"

#include <rdma/fi_cm.h> /* fi_listen */
#include <rdma/fi_endpoint.h>

#include <netinet/in.h> /* sockaddr_in */
#include <sys/select.h> /* fd_set, pselect */

#include <unistd.h> /* close */

#include <algorithm> /* max, move, transform */
#include <cassert>
#include <cerrno>
#include <chrono> /* seconds */
#include <functional> /* ref */
#include <memory> /* make_shared */
#include <stdexcept>
#include <string> /* to_string */
#include <thread> /* sleep_for */

Fabric_connection_server_factory::Fabric_connection_server_factory(Fabric &fabric_, event_producer &eq_, ::fi_info &info_, std::uint16_t port_)
  : _info(info_)
  , _fabric(fabric_)

  /*
   * "this" is context for events occuring on the passive endpoint.
   * Not really necessary, as the expected event so far is a CONNREQ
   * which is handled synchronously by server_control.
   */
  , _pep(fabric_.make_fid_pep(_info, this))
  /* register as an event consumer */
  , _event_registration(eq_, *this, *_pep)
  , _conn_info{nullptr}
  , _pending{}
  , _open{}
  , _end{}
  , _eq{eq_}
  , _th{&Fabric_connection_server_factory::listen, this, port_, _end.fd_read(), std::ref(*_pep), get_name(&_pep->fid)}
{
}

Fabric_connection_server_factory::~Fabric_connection_server_factory()
try
{
  char c{};
  auto sz = ::write(_end.fd_write(), &c, 1);
  (void) sz;
  _th.join();
}
catch ( const std::exception &e )
{
  std::cerr << "SERVER connection shutdown error " << e.what();
}

void *Fabric_connection_server_factory::query_interface(Component::uuid_t& itf_uuid) {
  return itf_uuid == IFabric_endpoint::iid() ? this : nullptr;
}

size_t Fabric_connection_server_factory::max_message_size() const
{
  return _info.ep_attr->max_msg_size;
}

std::string Fabric_connection_server_factory::get_provider_name() const
{
  return _info.fabric_attr->prov_name;
}

void Fabric_connection_server_factory::cb(std::uint32_t event, ::fi_eq_cm_entry &entry_) noexcept
try
{
  switch ( event )
  {
  case FI_CONNREQ:
    {
      auto conn = std::make_shared<Fabric_connection_server>(_fabric, _eq, *entry_.info);
      _pending.push(conn);
    }
    break;
  default:
    break;
  }
}
catch ( const std::exception &e )
{
  std::cerr << __func__ << " (Fabric_connection_server_factory) " << e.what() << "\n";
  throw;
}

void Fabric_connection_server_factory::err(::fi_eq_err_entry &) noexcept
{
  /* The passive endpoint receives an error. As it is not a connection request, ignore it. */
}

Fd_socket Fabric_connection_server_factory::make_listener(std::uint16_t port)
{
  constexpr int domain = AF_INET;
  Fd_socket fd(::socket(domain, SOCK_STREAM, 0));

  {
    int optval = 1;
    if ( -1 == ::setsockopt(fd.fd(), SOL_SOCKET, SO_REUSEADDR, &optval, sizeof optval) )
    {
      auto e = errno;
      system_fail(e, "setsockopt");
    }
  }

  {
    sockaddr_in addr{};
    addr.sin_family = domain;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    if ( -1 == ::bind(fd.fd(), pointer_cast<sockaddr>(&addr), sizeof addr) )
    {
      auto e = errno;
      system_fail(e, "bind for port " + std::to_string(port));
    }
  }

  if ( -1 == ::listen(fd.fd(), 10) )
  {
    auto e = errno;
    system_fail(e, "Server_control::make_listener ::listen ");
  }

  return fd;
}

/*
 * objects accessed in multiple threads:
 *
 *  fabric_ should be used only by (threadsafe) libfabric calls.
 *  info_ should be used only in by (threadsafe) libfabric calls.
 * _pending: has a mutex
 */
#include <iostream>
void Fabric_connection_server_factory::listen(
  std::uint16_t port_
  , int end_fd_
  , ::fid_pep &pep_
  , fabric_types::addr_ep_t pep_name_
)
{
  Fd_socket listen_fd(make_listener(port_));
  CHECK_FI_ERR(::fi_listen(&pep_));

  auto run = true;
  while ( run )
  {
    fd_set fds_read;
    FD_ZERO(&fds_read);
    FD_SET(listen_fd.fd(), &fds_read);
    FD_SET(_eq.fd(), &fds_read);
    FD_SET(end_fd_, &fds_read);

    auto n = ::pselect(std::max(std::max(_eq.fd(), listen_fd.fd()), end_fd_)+1, &fds_read, nullptr, nullptr, nullptr, nullptr);
    if ( n < 0 )
    {
      auto e = errno;
      switch ( e )
      {
      /* Cannot "fix" any of the error conditions, but do acknowledge their possibility */
      case EBADF:
      case EINTR:
      case EINVAL:
      case ENOMEM:
        break;
      default: /* unknown error */
        break;
      }
    }
    else
    {
      run = ! FD_ISSET(end_fd_, &fds_read);
      if ( FD_ISSET(listen_fd.fd(), &fds_read) )
      {
        try
        {
          auto r = ::accept(listen_fd.fd(), nullptr, nullptr);
          if ( r == -1 )
          {
            auto e = errno;
            system_fail(e, (" in accept fd " + std::to_string(listen_fd.fd())));
          }
          Fd_control conn_fd(r);
          /* NOTE: Fd_control needs a timeout. */

          /* we have a "control connection". Send the name of the server passive endpoint to the client */
          /* send the name to the client */
          conn_fd.send_name(pep_name_);
        }
        catch ( const std::exception &e )
        {
          std::cerr << "exception establishing connection: " << e.what() << "\n";
#if 1
          std::this_thread::sleep_for(std::chrono::seconds(1));
#else
          throw;
          /* An exception may not cause the loop to exit; only the destructor may do that. */
#endif
        }
      }
      if ( FD_ISSET(_eq.fd(), &fds_read) )
      {
        try
        {
          /* There may be something in the event queue. Go see what it is. */
          _eq.read_eq();
        }
        catch ( const std::exception &e )
        {
          std::cerr << "exception handling event queue: " << e.what() << "\n";
#if 1
          std::this_thread::sleep_for(std::chrono::seconds(1));
#else
          throw;
          /* An exception may not cause the loop to exit; only the destructor may do that. */
#endif
        }
      }
    }
  }
}

Component::IFabric_connection * Fabric_connection_server_factory::get_new_connections()
{
  Fabric_connection *r = nullptr;

  auto c = _pending.remove();
  if ( c )
  {
    c->expect_event(FI_CONNECTED);
    r = &*(c);
    _open.insert({r, c});
  }

  return r;
}
std::vector<Component::IFabric_connection*> Fabric_connection_server_factory::connections()
{
  std::vector<Fabric_connection *> v;
  std::transform(_open.begin(), _open.end(), std::back_inserter(v), [] (const open_t::value_type &v_) { return &*v_.second; });

  /* EXTRA COPY to change type */
  return std::vector<Component::IFabric_connection *>(v.begin(), v.end());
}

void Fabric_connection_server_factory::close_connection(Component::IFabric_connection * cnxn_)
{
  /*
   * If a server-side connection, remove it from the map.
   * If not in the map, presumably a client-side connection.
   * We do not own those.
   */
  auto it = _open.find(static_cast<Fabric_connection *>(cnxn_));
  if ( it != _open.end() )
  {
    _open.erase(it);
  }
}
