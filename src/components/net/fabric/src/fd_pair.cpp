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

#include "fd_pair.h"

#include "system_fail.h"

#include <unistd.h> /* close, pipe */
#include <fcntl.h> /* fcntl, O_NONBLOCK */

Fd_pair::Fd_pair()
  : _pair{}
{
  if ( -1 == ::pipe(_pair) )
  {
    auto e = errno;
    system_fail(e, "creating Fd_pair");
  }
  if ( -1 == ::fcntl(fd_read(), F_SETFL, O_NONBLOCK) )
  {
    auto e = errno;
    system_fail(e, "setting O_NONBLOCK on Fd_pair");
  }
}

Fd_pair::~Fd_pair()
{
  ::close(fd_read());
  ::close(fd_write());
}
