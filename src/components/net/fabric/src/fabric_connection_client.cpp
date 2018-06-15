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
#include "event_producer.h"
#include "fabric.h"
#include "fabric_check.h" /* CHECK_FI_ERR */
#include "fabric_error.h"
#include "fabric_str.h"
#include "fabric_types.h"
#include "system_fail.h"

#include <rdma/fi_cm.h> /* fi_connect, fi_shutdown */

#include <algorithm> /* copy */
#include <cerrno>
#include <cstdint> /* size_t */

#if 0
#include "duration_stat.h"
#include "timer_split.h"
#include "timer_to_exit.h"
#include "writer_at_exit.h"
using timer_split = toad::ut::timer_split;
using timer_to_exit = toad::ut::timer_to_exit;
using wr_dur = toad::ut::writer_at_exit<toad::ut::duration_stat>;
#endif

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
  fabric_types::addr_ep_t set_peer_early(std::unique_ptr<Fd_control> control_, ::fi_info &ep_info_)
  {
    fabric_types::addr_ep_t remote_addr;
    if ( ep_info_.ep_attr->type == FI_EP_MSG )
    {
      remote_addr = control_->recv_name();
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

  void fi_void_connect(::fid_ep &ep_, const ::fi_info &ep_info_, const void *addr_, const void *param_, size_t paramlen_)
  try
  {
    CHECK_FI_ERR(::fi_connect(&ep_, addr_, param_, paramlen_));
  }
  catch ( const fabric_error &e )
  {
    throw e.add(tostr(ep_info_));
  }
}

Fabric_connection_client::Fabric_connection_client(
  Fabric &fabric_
  , event_producer &ev_
  , ::fi_info &info_
  , const std::string & remote_
  , std::uint16_t control_port_
)
try
  : Fabric_connection(fabric_, ev_, info_, std::unique_ptr<Fd_control>(new Fd_control(remote_, control_port_)), set_peer_early)
  , _ev(ev_)
{
  if ( ep_info().ep_attr->type == FI_EP_MSG )
  {
    std::size_t paramlen = 0;
    auto param = nullptr;
    fi_void_connect(ep(), ep_info(), ep_info().dest_addr, param, paramlen);
    expect_event_sync(FI_CONNECTED);
  }
}
catch ( fabric_error &e )
{
  throw e.add("in Fabric_connection_client constuctor");
}

Fabric_connection_client::~Fabric_connection_client()
try
{
  /* "the flags parameter is reserved and must be 0" */
  ::fi_shutdown(&ep(), 0);
  /* The server may in turn give us a shutdown event. We do not need to see it. */
}
catch ( const std::exception &e )
{
  std::cerr << "CLIENT connection shutdown error " << e.what();
}

/* _ev.read_eq() in client, no-op in server */
void Fabric_connection_client::solicit_event() const
{
  _ev.read_eq();
}

void Fabric_connection_client::wait_event() const
{
  _ev.wait_eq();
}

void Fabric_connection_client::expect_event_sync(std::uint32_t event_exp) const
{
#if 0
  timer_split tm{};
  wr_dur ee0{std::cerr, "ee0"};
  wr_dur ee1{std::cerr, "ee1"};
  timer_to_exit tte{tm, ee1};
#endif
  ensure_event();
#if 0
  tte.split(ee0);
#endif
  expect_event(event_exp);
}
