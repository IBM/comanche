#ifndef _TEST_PINGPONG_SERVER_CLIENT_STATE_H_
#define _TEST_PINGPONG_SERVER_CLIENT_STATE_H_

#include "pingpong_cnxn_state.h" /* cnxn_state */
#include "server_connection.h"
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */

namespace Component
{
  class IFabric_server_factory;
}

struct client_state
{
  server_connection sc;
  cnxn_state st;
  explicit client_state(Component::IFabric_server_factory &factory_, std::size_t buffer_size_, std::uint64_t remote_key_, unsigned iteration_count_, std::size_t msg_size_);
  client_state(const client_state &) = delete;
  client_state(client_state &&cs_);
};

#endif
