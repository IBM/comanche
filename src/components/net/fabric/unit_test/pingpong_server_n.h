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
#ifndef _TEST_PINGPONG_SERVER_N_H_
#define _TEST_PINGPONG_SERVER_N_H_

#include <boost/core/noncopyable.hpp>

#include "pingpong_server_client_state.h"
#include "pingpong_stat.h"
#include <cstdint> /* uint64_t */
#include <thread>
#include <vector>

namespace Component
{
  class IFabric_server_factory;
}

/*
 * A Component::IFabric_server_factory
 */
class pingpong_server_n
  : private boost::noncopyable
{
  pingpong_stat _stat;
  std::thread _th;

  void listener(
    unsigned client_count
    , Component::IFabric_server_factory &factory
    , std::size_t buffer_size
    , std::uint64_t remote_key_base
    , unsigned iteration_count
    , std::size_t msg_size
  );
public:
  pingpong_server_n(
    unsigned client_count
    , Component::IFabric_server_factory &factory
    , std::size_t buffer_size
    , std::uint64_t remote_key_base
    , unsigned iteration_count
    , std::size_t msg_size
  );
  ~pingpong_server_n();
  pingpong_stat time();
};

#endif
