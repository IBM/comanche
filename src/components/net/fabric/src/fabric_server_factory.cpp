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

#include "fabric_server_factory.h"

#include "fabric_memory_control.h"
#include "fabric_op_control.h"
#include "fabric_server.h"

#include <algorithm> /* transform */
#include <iterator> /* back_inserter */
#include <memory> /* make_shared, static_pointer_cast */

Fabric_server_factory::Fabric_server_factory(Fabric &fabric_, event_producer &eq_, ::fi_info &info_, std::uint16_t port_)
  : Fabric_server_generic_factory(fabric_, eq_, info_,port_)
{}

Fabric_server_factory::~Fabric_server_factory()
{}

std::shared_ptr<Fabric_memory_control> Fabric_server_factory::new_server(Fabric &fabric_, event_producer &eq_, ::fi_info &info_)
{
  auto conn = std::make_shared<Fabric_server>(fabric_, eq_, info_);
  return
    std::static_pointer_cast<Fabric_memory_control>(
      std::static_pointer_cast<Fabric_op_control>(conn)
    );
}

Component::IFabric_server * Fabric_server_factory::get_new_connections()
{
  return static_cast<Fabric_server *>(Fabric_server_generic_factory::get_new_connection());
}

std::vector<Component::IFabric_server *> Fabric_server_factory::connections()
{
  auto g = Fabric_server_generic_factory::connections();
  std::vector<Component::IFabric_server *> v;
  std::transform(
    g.begin()
    , g.end()
    , std::back_inserter(v)
    , [] (Fabric_memory_control *v_)
      {
        return static_cast<Fabric_server *>(static_cast<Fabric_op_control *>(&*v_));
      }
  );
  return v;
}

void Fabric_server_factory::close_connection(Component::IFabric_server * cnxn_)
{
  return Fabric_server_generic_factory::close_connection(static_cast<Fabric_server *>(cnxn_));
}
