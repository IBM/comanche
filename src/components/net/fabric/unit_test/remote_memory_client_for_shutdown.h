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
#ifndef _TEST_REMOTE_MEMORY_FOR_SHUTDOWN_H_
#define _TEST_REMOTE_MEMORY_FOR_SHUTDOWN_H_

#include "remote_memory_client.h"

#include <cstdint> /* uint16_t, uint64_t */
#include <cstring> /* string */

namespace Component
{
  class IFabric;
}

class remote_memory_client_for_shutdown
  : private remote_memory_client
{
public:
  remote_memory_client_for_shutdown(
    Component::IFabric &fabric
    , const std::string &fabric_spec
    , const std::string ip_address
    , std::uint16_t port
    , std::size_t memory_size
    , std::uint64_t remote_key_base
  );
  using remote_memory_client::max_message_size;
};

#endif
