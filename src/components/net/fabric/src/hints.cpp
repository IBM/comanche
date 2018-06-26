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

#include "hints.h"

#include "fabric_util.h" /* make_fi_info */

#include <cassert>
#include <cstring> /* strdup */

/**
 * Fabric/RDMA-based network component
 *
 */

hints::hints()
  : _info(make_fi_info())
{}

hints::hints(std::shared_ptr<fi_info> info_)
  : _info(info_)
{}

hints &hints::caps(uint64_t c) { _info->caps = c; return *this; }

hints &hints::mode(uint64_t c) { _info->mode = c; return *this; }

hints &hints::mr_mode(int m) { _info->domain_attr->mr_mode = m; return *this; }

hints &hints::prov_name(const char *n)
{
  assert(! _info->fabric_attr->prov_name);
  _info->fabric_attr->prov_name = ::strdup(n); return *this;
}

const char *hints::prov_name() const { return _info->fabric_attr->prov_name; }

const fi_info &hints::data() { return *_info; }
