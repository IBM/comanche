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

#include "fd_socket.h"

#include "system_fail.h"

#include <netinet/in.h>
#include <sys/socket.h>

#include <unistd.h>

#include <cerrno>
#include <stdexcept>

Fd_socket::Fd_socket()
  : _fd(-1)
{}

Fd_socket::Fd_socket(int fd_)
  : _fd(fd_)
{
  if ( _fd < 0 )
  {
    throw std::logic_error("negative fd in Fd_socket::Fd_socket");
  }
}

Fd_socket::~Fd_socket()
{
  close();
}

Fd_socket::Fd_socket(Fd_socket &&o) noexcept
  : _fd(o._fd)
{
  o._fd = -1;
}

Fd_socket &Fd_socket::operator=(Fd_socket &&o) noexcept
{
  if ( this != &o )
  {
    close();
    _fd = o._fd;
    o._fd = -1;
  }
  return *this;
}

void Fd_socket::close() noexcept
{
  if ( _fd != -1 )
  {
    ::close(_fd);
  }
}

void Fd_socket::send(const void *buf, std::size_t size) const
{
  auto r = ::send(_fd, buf, size, MSG_NOSIGNAL);
  if ( r < 0 )
  {
    auto e = errno;
    system_fail(e, "send");
  }
  if ( r == 0 )
  {
     system_fail(ECONNABORTED, "send");
  }
}

void Fd_socket::recv(void *buf, std::size_t size) const
{
  ssize_t r;
  do
  {
     r = ::read(_fd, buf, size);
  } while (r == -1 && ( errno == EAGAIN || errno == EWOULDBLOCK ) );
  if ( r < 0 )
  {
    auto e = errno;
    system_fail(e, "recv (neg) ");
  }
  if ( r == 0 )
  {
    system_fail(ECONNABORTED, "recv (zero) ");
  }
}
