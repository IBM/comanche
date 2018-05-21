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

#include "fabric_connection_server.h"

#include "fabric_util.h"

#include <rdma/fi_cm.h>

#include <algorithm>
#include <stdexcept>
#if 0
class bad_dest_addr_alloc
  : public std::bad_alloc
{
  std::string _what;
public:
  explicit bad_dest_addr_alloc(std::size_t sz)
    : _what{"Failed to allocate " + std::to_string(sz) + " bytes for dest_addr"}
  {}
  const char *what() const noexcept { return _what.c_str(); }
};
#endif
namespace
{
  /*
   * Callback for verbs behavior on client side; a no-op on the server side;
   */
  addr_ep_t set_peer_early(Fd_control &, fi_info &)
  {
    return addr_ep_t{};
  }
}

Fabric_connection_server::Fabric_connection_server(
  fid_fabric & fabric_
  , fid_eq &eq_
  , fi_info &info_
  , Fd_control &&conn_fd_
)
  : Fabric_connection(fabric_, eq_, info_, std::move(conn_fd_), set_peer_early, "server")
{
  if ( ep_info().ep_attr->type == FI_EP_MSG )
  {
    std::size_t paramlen = 0;
    auto param = nullptr;
    CHECKZ(fi_accept(&ep(), param, paramlen));

    await_connected(eq_);
  }
}
