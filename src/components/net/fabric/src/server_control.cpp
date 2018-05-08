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

#include "server_control.h"

#include "fabric_connection.h"
#include "fabric_help.h"
#include "fabric_json.h"
#include "fd_control.h"
#include "fd_socket.h"
#include "pointer_cast.h"
#include "system_fail.h"

#include <rdma/fabric.h> /* fi_info */

#include <netinet/in.h>
#include <sys/socket.h>

#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <stdexcept>

Server_control::Server_control(fid_fabric &fabric_, const fi_info &info_, std::uint16_t port_)
  : _pending{}
  , _open{}
  , _end{}
  , _pipe_errno(::pipe(_end) == -1 ? errno : 0)
  , _th{&Server_control::listen, make_listener(port_), _end[0], std::ref(fabric_), std::ref(info_), std::ref(_pending)}
{
  if ( _pipe_errno )
  {
    system_fail(_pipe_errno, "creating server control pipe");
  }
}

Server_control::~Server_control()
{
  ::close(_end[1]);
  _th.join();
}

Fd_socket Server_control::make_listener(std::uint16_t port)
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
      system_fail(e, "bind");
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
 * objects accesses in multiple threads:
 * 
 *  fabric_ should be used only by (threadsafe) libfabric calls.
 *  info_ should be used only in by (threadsafe) libfabric calls.
 *  _run : needs an atomic
 *  _pending: needs a mutex 
 */
#include <cassert>
#include <cstring>
#include <unistd.h>
#include <sys/select.h>
void Server_control::listen(Fd_socket &&listen_fd_, int end_fd_, fid_fabric &fabric_, const fi_info &info_, Pending_cnxns &pend_)
//void Server_control::listen(Fd_socket &&listen_fd_, int end_fd_, fid_fabric &fabric_, const fi_info &info_) 
{
  auto listen_fd = std::move(listen_fd_);

  auto run = true;
  while ( run )
  { 
    fd_set fds_read;
    FD_ZERO(&fds_read);
    FD_SET(listen_fd.fd(), &fds_read);
    FD_SET(end_fd_, &fds_read);
#if 0
    auto ready =
#endif
    ::pselect(std::max(listen_fd.fd(), end_fd_)+1, &fds_read, nullptr, nullptr, nullptr, nullptr);

#if 0
    if ( -1 == ready )
    {
      auto e = errno;
    }
#endif

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
        auto conn_fd = Fd_control(r);

        /* we have a "control connection". Create a fabric endpoint. */
        /* send format, so that the client can create an endpoint with the servers address format */
        conn_fd.send_format(Fd_control::format_ep_t(info_.addr_format));
        auto conn = std::make_shared<Fabric_connection>(fabric_, info_, std::move(conn_fd), false);
        /* get the address format and name for the server endpoint */
        auto ep_name = conn->get_name();

        /* send the format and name to the client */
        conn->control().send_name(ep_name);
        pend_.push(conn);
      }
      catch ( const std::exception &e )
      {
        /* An exception may not cause the loop to exit; only the destructor may do that. */
        sleep(1);
      }
    }
  }
  ::close(end_fd_);
}

Fabric_connection * Server_control::get_new_connection()
{
  Fabric_connection *r = nullptr;

  auto c = _pending.remove();
  if ( c )
  {
    r = &*(c);
    _open.insert({r, c});
  }

  return r;
}

std::vector<Fabric_connection *> Server_control::connections()
{
  std::vector<Fabric_connection *> v;
  std::transform(_open.begin(), _open.end(), std::back_inserter(v), [] (const open_t::value_type &v) { return &*v.second; });
  return v;
}

void Server_control::close_connection(Fabric_connection * cnxn_)
{
  /*
   * If a server-side connection, remove it from the map.
   * If not in the map, presumable a client-side connection.
   * We do not own those, so delete it.
   */
  auto it = _open.find(cnxn_);
  if ( it != _open.end() )
  {
    _open.erase(it);
  }
  else
  {
    delete cnxn_;
  }
}
