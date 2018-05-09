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

#include <cstdint> /* uint64_t */
#include <stdexcept>

struct fi_info;
struct fid_fabric;
struct fid_cq;
struct fid_ep;
struct fid_mr;

class Fabric_factory;

class Fabric_connection_factory
  : public Component::IFabric_endpoint
{
  std::shared_ptr<fi_info> _info;
  fid_fabric &_fabric;
  Server_control _control;
  std::vector<char> get_name(int (*getter)(fid_t fid, void *addr, size_t *addrlen));
protected:
  static constexpr std::uint64_t control_port = 47591;
public:
  // Fabric_endpoint_active(Fabric_factory & fabric, fi_info &info, uint64_t flags);
  Fabric_connection_factory(fid_fabric &fabric, const fi_info &info, std::uint16_t control_port);

  void *query_interface(Component::uuid_t& itf_uuid) override;

  Component::IFabric_connection * connect(const std::string& remote_endpoint) override;

  Component::IFabric_connection * get_new_connections() override;
 
  void close_connection(Component::IFabric_connection * connection) override;

  std::vector<Component::IFabric_connection*> connections() override;
 
  size_t max_message_size() const override;

  std::string get_provider_name() const override;
};

#endif
