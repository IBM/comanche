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

#ifndef _FABRIC_CONNECTION_FACTORY_H_
#define _FABRIC_CONNECTION_FACTORY_H_

#include <api/fabric_itf.h>

#include "fabric_ptr.h" /* fid_unique_ptr */
#include "server_control.h"

#include <cstdint> /* uint16_t */
#include <memory> /* shared_ptr */

struct fi_info;
struct fid_fabric;
struct fid_eq;
struct fid_pep;

class Fabric_factory;

class Fabric_connection_factory
  : public Component::IFabric_endpoint
{
  fi_info &_info;
  fid_fabric &_fabric;
  fid_eq &_eq;
  std::shared_ptr<fid_pep> _pep;
  Server_control _control;
public:
  /* Note: fi_info is not const because we reuse it when constructing the passize endpoint */
  Fabric_connection_factory(fid_fabric &fabric, fid_eq &eq, fi_info &info, std::uint16_t control_port);

  void *query_interface(Component::uuid_t& itf_uuid) override;

  Component::IFabric_connection * get_new_connections() override;

  void close_connection(Component::IFabric_connection * connection) override;

  std::vector<Component::IFabric_connection*> connections() override;

  std::size_t max_message_size() const override;

  std::string get_provider_name() const override;
};

#endif
