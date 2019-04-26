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


#ifndef _FABRIC_FACTORY_H_
#define _FABRIC_FACTORY_H_

#include <api/fabric_itf.h>

#include <component/base.h> /* DECLARE_VERSION, DECLARE_COMPONENT_UUID */

/*
 * Note: Fabric_factory make fabrics as specified by configuration parameters
 * (and limited to those fabrics supported by the available software and hardware).
 */
class Fabric_factory
  : public Component::IFabric_factory
{
public:
  DECLARE_VERSION(0.1f);
  DECLARE_COMPONENT_UUID(0xfac3a5ae,0xcf34,0x4aff,0x8321,0x19,0x08,0x21,0xa9,0x9f,0xd3);
  void *query_interface(Component::uuid_t& itf_uuid) override;

  Fabric_factory();
  /**
   * Open a fabric provider instance
   *
   * @param json_configuration Configuration string in JSON
   * form. e.g. {
   *   "caps":["FI_MSG","FI_RMA"],
   *   "fabric_attr": { "prov_name" : "verbs" },
   *   "bootstrap_addr":"10.0.0.1:9999" }
   * @return
   *
   * caps:
   * prov_name: same format as fi_fabric_attr::prov_name
   *
   * @throw fabric_bad_alloc : std::bad_alloc - out of memory
   * @throw std::domain_error : json file parse-detected error
   * @throw fabric_runtime_error : std::runtime_error : ::fi_control fail
   */
  Component::IFabric * make_fabric(const std::string& json_configuration) override;
};

#endif
