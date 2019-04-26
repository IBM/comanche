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


#include "fabric_connection_server.h"

#include "fabric_check.h" /* CHECK_FI_ERR */
#include "fd_control.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fi_cm.h> /* fi_accept, fi_shutdown */
#pragma GCC diagnostic pop

#include <cstdint> /* size_t */
#include <exception>
#include <iostream> /* cerr */
#include <memory> /* unique_ptr */

namespace
{
  /*
   * Callback for verbs behavior on client side; a no-op on the server side;
   */
  fabric_types::addr_ep_t set_peer_early(std::unique_ptr<Fd_control>, ::fi_info &)
  {
    return fabric_types::addr_ep_t{};
  }
}

Fabric_connection_server::Fabric_connection_server(
  Fabric &fabric_
  , event_producer &ev_
  , ::fi_info &info_
)
  : Fabric_op_control(fabric_, ev_, info_, std::unique_ptr<Fd_control>(), set_peer_early)
{
  if ( ep_info().ep_attr->type == FI_EP_MSG )
  {
    std::size_t paramlen = 0;
    auto param = nullptr;
    CHECK_FI_ERR(::fi_accept(&ep(), param, paramlen));
  }
}

Fabric_connection_server::~Fabric_connection_server()
try
{
  /* "the flags parameter is reserved and must be 0" */
  ::fi_shutdown(&ep(), 0);
/* The client may in turn call fi_shutdown, giving us an event. We do not need to see it.
 */
}
catch ( const std::exception &e )
{
  std::cerr << "SERVER connection shutdown error " << e.what();
}

/* The server does not need to do anything to solicit an event,
 * as the server_factory continuously reads the server's event queue
 */
void Fabric_connection_server::solicit_event() const
{
}

void Fabric_connection_server::wait_event() const
{
}
