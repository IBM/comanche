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

#ifndef _FABRIC_FACTORY_H_
#define _FABRIC_FACTORY_H_

#include <api/fabric_itf.h>

#include <component/base.h> /* DECLARE_VERSION, DECLARE_COMPONENT_UUID */

#include <memory> /* shared_ptr */

struct fi_info;
struct fid_fabric;

/*
 * Note: Fabric_factory is a fabric which makes passive endpoints and domains; it does not make fabrics.
 */
class Fabric_factory
  : public Component::IFabric_factory
{
  std::shared_ptr<fi_info> _fabric_spec;
  std::shared_ptr<fi_info> _info; 
  std::shared_ptr<fid_fabric> _fabric;
public:
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac3a5ae,0xcf34,0x4aff,0x8321,0x19,0x08,0x21,0xa9,0x9f,0xd3);
  void *query_interface(Component::uuid_t& itf_uuid) override;

  Fabric_factory(const std::string& json_configuration);

  Component::IFabric_endpoint * open_endpoint(const std::string& json_configuration) override;
  fid_fabric *fid() const { return &*_fabric; }
  std::string prov_name() const;
};

#endif
