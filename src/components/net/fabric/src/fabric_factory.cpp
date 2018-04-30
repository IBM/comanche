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

/* 
 * Authors: 
 * 
 */

#include "fabric_factory.h"

#include "fabric_endpoint.h"
#include "fabric_error.h"
#include "fabric_help.h"
#include "fabric_json.h"

#include <rdma/fabric.h> /* fi_info */

/** 
 * Fabric/RDMA-based network component
 * 
 */

/** 
 * Open a fabric provider instance
 * 
 * @param json_configuration Configuration string in JSON
 * form. e.g. {
 *   "caps":["FI_MSG","FI_RMA"],
 *   "preferred_provider" : "verbs",
 *   "bootstrap_addr":"10.0.0.1:9999" }
 * @return 
 *
 * caps: 
 * preferred_provider: same format as struct fi_fabric_attr::prov_name
 */

Fabric_factory::Fabric_factory(const std::string& json_configuration)
  : _fabric_spec(make_fi_fabric_spec(json_configuration)) 
  , _info(make_fi_info(*_fabric_spec))
  , _fabric(make_fid_fabric(*_info->fabric_attr, this))
{
}

void *Fabric_factory::query_interface(Component::uuid_t& itf_uuid) {
  return itf_uuid == IFabric_factory::iid() ? this : nullptr;
}

Component::IFabric_endpoint * Fabric_factory::open_endpoint(const std::string& json_configuration)
{
  _info = parse_info(_info, json_configuration);
  return new Fabric_endpoint(*this, *_info);
}

Component::IFabric_connection * Fabric_factory::open_connection(const std::string & json_configuration, const std::string & remote_endpoint)
{
  not_implemented(__func__);
}

std::string Fabric_factory::prov_name() const
{
  return _info->fabric_attr->prov_name;
}

/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  return component_id == Fabric_factory::component_id() ? new Fabric_factory("{\"caps\":[\"FI_RMA\"]}") : nullptr;
}
