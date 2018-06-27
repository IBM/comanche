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

#ifndef _FABRIC_SERVER_FACTORY_H_
#define _FABRIC_SERVER_FACTORY_H_

#include <api/fabric_itf.h> /* Component::IFabric_server_factory */
#include "fabric_server_generic_factory.h"

#include <cstdint> /* uint16_t */

struct fi_info;

class Fabric;
class event_producer;

class Fabric_server_factory
  : public Component::IFabric_server_factory
  , public Fabric_server_generic_factory
{
public:
  /* Note: fi_info is not const because we reuse it when constructing the passize endpoint */
  explicit Fabric_server_factory(Fabric &fabric, event_producer &ev_pr, ::fi_info &info, std::uint16_t control_port);
  Fabric_server_factory(const Fabric_server_factory &) = delete;
  Fabric_server_factory& operator=(const Fabric_server_factory &) = delete;
  ~Fabric_server_factory();

  Component::IFabric_server* get_new_connections() override;

  void close_connection(Component::IFabric_server* connection) override;

  std::vector<Component::IFabric_server*> connections() override;

  std::shared_ptr<Fabric_memory_control> new_server(Fabric &fabric, event_producer &eq, ::fi_info &info) override;

  std::size_t max_message_size() const override { return Fabric_server_generic_factory::max_message_size(); }
  std::string get_provider_name() const override { return Fabric_server_generic_factory::get_provider_name(); }

  void cb(std::uint32_t event, ::fi_eq_cm_entry &entry) noexcept override { return Fabric_server_generic_factory::cb(event, entry); }
  void err(::fi_eq_err_entry &entry) noexcept override { return Fabric_server_generic_factory::err(entry); }
};

#endif
