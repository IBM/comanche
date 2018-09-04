#ifndef _TEST_PINGPONG_SERVER_CLIENT_STATE_H_
#define _TEST_PINGPONG_SERVER_CLIENT_STATE_H_

#include "pingpong_cnxn_state.h" /* cnxn_state */
#include "server_connection.h"
#include <cstddef> /* size_t */

namespace Component
{
  class IFabric_server_factory;
}

struct client_state
{
  server_connection sc;
  cnxn_state st;
  explicit client_state(
    Component::IFabric_server_factory &factory
    , unsigned iteration_count
    , std::size_t msg_size
  );
  client_state(const client_state &) = delete;
  ~client_state();
};

#endif
