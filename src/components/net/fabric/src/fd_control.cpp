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

#include "fd_control.h"

#include "system_fail.h"

#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>

#include <memory>

namespace
{
  std::shared_ptr<addrinfo> getaddrinfo_ptr(std::string dst_addr, uint16_t port)
  {
    addrinfo hints{};
  
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_NUMERICSERV;

    addrinfo *presults;
    auto r = ::getaddrinfo(dst_addr.c_str(), std::to_string(port).c_str(), &hints, &presults);
    if ( r )
    {
      system_fail(r, __func__);
    }
    return std::shared_ptr<addrinfo>(presults, ::freeaddrinfo);
  }
}

Fd_control::Fd_control()
 : Fd_socket()
{}

Fd_control::Fd_control(int fd_)
 : Fd_socket(fd_)
{}

Fd_control::Fd_control(std::string dst_addr, uint16_t port)
  : Fd_socket()
{
  auto results = getaddrinfo_ptr(dst_addr, port);
  auto e = ENOENT;

  Fd_control connfd;
  for ( auto rp = results.get(); rp && ! good(); rp = rp->ai_next)
  {
    auto fd = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
    e = errno;
    if ( fd != -1 )
    {
      if ( -1 == ::connect(fd, rp->ai_addr, rp->ai_addrlen) )
      {
        e = errno;
      }
      else
      {
        *this = Fd_control(fd);
      }
    }
  }

  if ( ! good() )
  {
    sleep(1);
    system_fail(e, __func__);
  }
}

void Fd_control::send_format(const format_ep_t &format_) const
{
  auto nfmt = htonl(std::get<0>(format_));
  send(&nfmt, sizeof nfmt);
}

void Fd_control::send_name(const addr_ep_t &name_) const
{
  auto sz = std::get<0>(name_).size();
  auto nsz = htonl(sz);
  send(&nsz, sizeof nsz);
  send(&*std::get<0>(name_).begin(), sz);
}

auto Fd_control::recv_format() const -> format_ep_t
try
{
  auto nfmt = htonl((typename std::tuple_element<0, format_ep_t>::type()));
  recv(&nfmt, sizeof nfmt);
  return format_ep_t(ntohl(nfmt));
}
catch ( const std::exception &e )
{
  throw std::runtime_error(std::string("(while receiving format) ") + e.what());
}

auto Fd_control::recv_name() const -> addr_ep_t
try
{
  auto nsz = htonl(0);
  recv(&nsz, sizeof nsz);

  auto sz = ntohl(nsz);
  std::vector<char> name(sz);
  recv(&*name.begin(), sizeof sz);

  return addr_ep_t(std::move(name));
}
catch ( const std::exception &e )
{
  throw std::runtime_error(std::string("(while receiving name) ") + e.what());
}
