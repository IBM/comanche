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

#include "fabric_connection_factory.h"

#include "fabric_connection.h"
#include "fabric_error.h"
#include "fabric_util.h"
#include "fd_socket.h"

#include <netdb.h>

#include <memory> /* allocator, shared_ptr */
#include <stdexcept>

#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

Fabric_connection_factory::Fabric_connection_factory(fid_fabric &fabric_, fid_eq &eq_, fi_info &info_, std::uint16_t port_)
  : _info(info_)
  , _fabric(fabric_)
  , _eq(eq_)
  /* "this" is context for events occuring on the passive endpoint.
   * Not really necessary, as the expected event so far is a CONNREQ
   * which is handled synchronously by server_control.
   */
  , _pep(make_fid_pep_listener(fabric_, _info, _eq, this))
  , _control(_fabric, _eq, get_name(&_pep->fid), port_)
{
}

void *Fabric_connection_factory::query_interface(Component::uuid_t& itf_uuid) {
  return itf_uuid == IFabric_endpoint::iid() ? this : nullptr;
}

size_t Fabric_connection_factory::max_message_size() const
{
  return _info.ep_attr->max_msg_size;
}

std::string Fabric_connection_factory::get_provider_name() const
{
  return _info.fabric_attr->prov_name;
}

Component::IFabric_connection * Fabric_connection_factory::get_new_connections()
{
  return _control.get_new_connection();
}

std::vector<Component::IFabric_connection*> Fabric_connection_factory::connections()
{
  auto v = _control.connections();
  /* EXTRA COPY to change type */
  return std::vector<Component::IFabric_connection*>(v.begin(), v.end());
}

void Fabric_connection_factory::close_connection(Component::IFabric_connection * connection)
{
  if ( auto cnxn = dynamic_cast<Fabric_connection *>(connection) )
  {
    _control.close_connection(cnxn);
  }
}
