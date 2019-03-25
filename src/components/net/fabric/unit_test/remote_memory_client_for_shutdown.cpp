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
#include "remote_memory_client_for_shutdown.h"

remote_memory_client_for_shutdown::remote_memory_client_for_shutdown(
  Component::IFabric &fabric_
  , const std::string &fabric_spec_
  , const std::string ip_address_
  , std::uint16_t port_
  , std::uint64_t memory_size_
  , std::uint64_t remote_key_base_
)
  : remote_memory_client(fabric_, fabric_spec_, ip_address_, port_, memory_size_, remote_key_base_)
{
  do_quit();
}
