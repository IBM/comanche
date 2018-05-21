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

#ifndef _FABRIC_H_
#define _FABRIC_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <api/fabric_itf.h>
#pragma GCC diagnostic pop

#include <rdma/fi_domain.h>

#include <memory> /* shared_ptr */

struct fi_info;
struct fid_fabric;
struct fid_eq;

/*
 * Note: Fabric is a fabric which can create servers (IFabric_endpoint) and clients (IFabric_connection)
 */
class Fabric
  : public Component::IFabric
{
  std::shared_ptr<fi_info> _info;
  std::shared_ptr<fid_fabric> _fabric;
  /* an event queue, in case the endpoint is connection-oriented */
  fi_eq_attr _eq_attr;
  std::shared_ptr<fid_eq> _eq;
public:

  Fabric(const std::string& json_configuration);
  Component::IFabric_endpoint * open_endpoint(const std::string& json_configuration, std::uint16_t control_port) override;
  Component::IFabric_connection * open_connection(const std::string& json_configuration, const std::string & remote, std::uint16_t control_port) override;
};

#endif
