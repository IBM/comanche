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

#include "fabric_connection_client.h"

#include "bad_dest_addr_alloc.h"
#include "fabric_error.h"
#include "fabric_types.h"
#include "fabric_util.h"

#include <algorithm> /* copy */
#include <cstdint> /* size_t */

namespace
{
  /*
   * Establish the control port early because the verbs provider ignores
   * the addr parameter of fi_connect and uses only the dest_addr from the hints
   * provided when the endpoint was created.
   *
   * (This is a work-around for what looks like a bug in the verbs provider.
   * It should probably accept addr, as the sockets provider does.)
   */
  fabric_types::addr_ep_t set_peer_early(Fd_control &control_, fi_info &ep_info_)
  {
    fabric_types::addr_ep_t remote_addr;
    if ( ep_info_.ep_attr->type == FI_EP_MSG )
    {
      remote_addr = control_.recv_name();
      /* fi_connect, at least for verbs, ignores addr and uses dest_addr from the hints. */
      ep_info_.dest_addrlen = remote_addr.size();
      if ( 0 != ep_info_.dest_addrlen )
      {
        ep_info_.dest_addr = malloc(ep_info_.dest_addrlen);
        if ( ! ep_info_.dest_addr )
        {
          throw bad_dest_addr_alloc(ep_info_.dest_addrlen);
        }
        std::copy(remote_addr.begin(), remote_addr.end(), static_cast<char *>(ep_info_.dest_addr));
      }
    }
    /* Other providers will look in addr: provide the name there as well. */
    return remote_addr;
  }
}

Fabric_connection_client::Fabric_connection_client(
  fid_fabric & fabric_
  , fid_eq &eq_
  , fi_info &info_
  , const std::string & remote_
  , std::uint16_t control_port_
)
try
  : Fabric_connection(fabric_, eq_, info_, Fd_control(remote_, control_port_), set_peer_early)
{
  if ( ep_info().ep_attr->type == FI_EP_MSG )
  {
    std::size_t paramlen = 0;
    auto param = nullptr;
    fi_void_connect(ep(), ep_info(), ep_info().dest_addr, param, paramlen);

    await_connected(eq_);
  }
}
catch ( fabric_error &e )
{
  throw e.add("in Fabric_connection_client constuctor");
}
