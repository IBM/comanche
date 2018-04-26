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

#include <rapidjson/document.h>

#include <rdma/fabric.h> /* fi_info */

#include <cstdint> /* uint64_t */
#include <stdexcept> /* domain_error */

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

#include <rapidjson/document.h>

#include <cstdint>

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
  rapidjson::Document jdoc;
  jdoc.Parse(json_configuration.c_str());
#if 0
  auto provider = jdoc.FindMember("preferred_provider");
  auto provider_str = std::string(provider != jdoc.MemberEnd() && provider->value.IsString() ? provider->value.GetString() : "verbs");
#endif
  std::uint64_t caps_int{0U};
  auto caps = jdoc.FindMember("caps");
  if ( caps != jdoc.MemberEnd() && caps->value.IsArray() )
  {
    for ( auto cap = caps->value.Begin(); cap != caps->value.End(); ++cap )
    {
      if ( cap->IsString() )
      {
        auto cap_int_i = cap_fwd.find(cap->GetString());
        if ( cap_int_i != cap_fwd.end() )
        {
          caps_int |= cap_int_i->second;
        }
        else {
          throw std::domain_error(std::string("No such capability ") + cap->GetString());
        }
      }
    }
  }

#if 0
  assert(! _info->fabric_attr->prov_name);
  _info->fabric_attr->prov_name = ::strdup(provider_str.c_str());
#endif

  return new Fabric_endpoint(*this, *_info);
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
