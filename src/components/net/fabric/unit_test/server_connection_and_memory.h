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
#ifndef _TEST_SERVER_CONNECTION_AND_MEMORY_H_
#define _TEST_SERVER_CONNECTION_AND_MEMORY_H_

#include "server_connection.h"
#include "registered_memory.h"
#include "remote_memory_accessor.h"
#include <boost/core/noncopyable.hpp>
#include <cstdint> /* uint64_t */

namespace Component
{
  class IFabric_server_factory;
}

class server_connection_and_memory
  : public server_connection
  , public registered_memory
  , public remote_memory_accessor
  , private boost::noncopyable
{
public:
  server_connection_and_memory(
    Component::IFabric_server_factory &ep
    , std::size_t memory_size
    , std::uint64_t remote_key
  );
  ~server_connection_and_memory();
};

#endif
