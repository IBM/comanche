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

#include "fabric_runtime_error.h"

/*
 * Authors:
 *
 */

#include <rdma/fi_errno.h>

#include <string>

Component::IFabric_runtime_error::IFabric_runtime_error(const std::string &what_)
  : std::runtime_error(what_)
{
}

/**
 * Fabric/RDMA-based network component
 */

fabric_runtime_error::fabric_runtime_error(unsigned i_, const char *file_, int line_)
  : Component::IFabric_runtime_error{std::string{"fabric_runtime_error \""} + ::fi_strerror(int(i_)) + "\" at " + file_ + ":" + std::to_string(line_)}
  , _i(i_)
  , _file(file_)
  , _line(line_)
{}

fabric_runtime_error::fabric_runtime_error(unsigned i_, const char *file_, int line_, const std::string &desc_)
  : Component::IFabric_runtime_error{std::string{"fabric_runtime_error ("} + std::to_string(i_) + ") \"" + ::fi_strerror(int(i_)) + "\" at " + file_ + ":" + std::to_string(line_) + " " + desc_}
  , _i(i_)
  , _file(file_)
  , _line(line_)
{}

fabric_runtime_error fabric_runtime_error::add(const std::string &added_) const
{
  return fabric_runtime_error(_i, _file, _line, added_);
}
