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
#ifndef _DAWN_HSTORE_TEST_STORE_MAP_H_
#define _DAWN_HSTORE_TEST_STORE_MAP_H_

#include <api/components.h>

#include <string>
#include <map>

// The fixture for testing class Foo.
class store_map
{
public:
  struct impl_spec
  {
    std::string       name;
    Component::uuid_t factory_id;
  };

  using impl_map_t = std::map<std::string, impl_spec>;
private:
  static const std::string impl_default;
  static const impl_map_t impl_map;
  static const impl_map_t::const_iterator impl_env_it;
  static const impl_spec pmstore_impl;
public:
  static const impl_spec *const impl;
  static const std::string location;
  static std::string numa_zone()
  {
    return "0";
  }
};

#endif
