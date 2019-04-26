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
#ifndef _TEST_PATIENCE_H_
#define _TEST_PATIENCE_H_

#include <cstdint> /* uint16_t */
#include <string>

namespace Component
{
  class IFabric;
  class IFabric_client;
  class IFabric_client_grouped;
}

Component::IFabric_client *open_connection_patiently(Component::IFabric &fabric, const std::string &fabric_spec, const std::string ip_address, std::uint16_t port);

Component::IFabric_client_grouped *open_connection_grouped_patiently(Component::IFabric &fabric, const std::string &fabric_spec, const std::string ip_address, std::uint16_t port);

#endif
