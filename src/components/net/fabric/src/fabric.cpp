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

#include "fabric.h"

#include "fabric_connection_factory.h"
#include "fabric_connection_client.h"
#include "fabric_error.h"
#include "fabric_json.h"
#include "fabric_util.h"

#include <rdma/fabric.h> /* fi_info */

#include <stdexcept> /* system_error */

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
 *   "fabric_attr": { "prov_name" : "verbs" },
 *   "bootstrap_addr":"10.0.0.1:9999" }
 * @return
 *
 * caps:
 * prov_name: same format as fi_fabric_attr::prov_name
 */

namespace
{
  fi_eq_attr &eq_attr_init(fi_eq_attr &attr_)
  {
    attr_.size = 10;
    attr_.wait_obj = FI_WAIT_NONE;
    return attr_;
  }
}

Fabric::Fabric(const std::string& json_configuration)
  : _info(make_fi_info(json_configuration))
  , _fabric(make_fid_fabric(*_info->fabric_attr, this))
  , _eq_attr{}
  , _eq(make_fid_eq(*_fabric, eq_attr_init(_eq_attr), this))
{
}

Component::IFabric_endpoint * Fabric::open_endpoint(const std::string& json_configuration, std::uint16_t control_port_)
{
  _info = parse_info(json_configuration, _info);
  return new Fabric_connection_factory(*_fabric, *_eq, *_info, control_port_);
}

namespace
{
  std::string while_in(const std::string &where)
  {
    return " (while in " + where + ")";
  }
}

Component::IFabric_connection * Fabric::open_connection(const std::string& json_configuration_, const std::string &remote_, std::uint16_t control_port_)
try
{
  _info = parse_info(json_configuration_, _info);
  return new Fabric_connection_client(*_fabric, *_eq, *_info, remote_, control_port_);
}
catch ( const fabric_error &e )
{
  throw e.add(while_in(__func__));
}
catch ( const std::system_error &e )
{
  throw std::system_error(e.code(), e.what() + while_in(__func__));
}
