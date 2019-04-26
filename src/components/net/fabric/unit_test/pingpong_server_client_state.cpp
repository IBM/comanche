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
#include "pingpong_server_client_state.h"

#include <api/fabric_itf.h>

client_state::client_state(
  Component::IFabric_server_factory &factory_
  , unsigned iteration_count_
  , std::size_t msg_size_
)
  : sc(factory_)
  , st(
    sc.cnxn()
    , iteration_count_
    , msg_size_
  )
{
}

client_state::~client_state()
{
}

