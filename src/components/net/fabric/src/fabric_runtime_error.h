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


#ifndef _FABRIC_ERROR_H_
#define _FABRIC_ERROR_H_

/*
 * Authors:
 *
 */

#include <api/fabric_itf.h>

#include <stdexcept>

#include <string>

/**
 * Fabric/RDMA-based network component
 */

class fabric_runtime_error
  : public Component::IFabric_runtime_error
{
  unsigned _i;
  const char *_file;
  int _line;
public:
  explicit fabric_runtime_error(unsigned i, const char *file, int line);
  explicit fabric_runtime_error(unsigned i, const char *file, int line, const std::string &desc);
  fabric_runtime_error(const fabric_runtime_error &) = default;
  fabric_runtime_error& operator=(const fabric_runtime_error &) = default;
  fabric_runtime_error add(const std::string &added) const;
  unsigned id() const noexcept { return _i; }
};

#endif
