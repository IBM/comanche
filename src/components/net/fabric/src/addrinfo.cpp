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

#include "addrinfo.h"

#include "system_fail.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include <string> /* to_string */

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
