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
