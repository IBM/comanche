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

#ifndef _OPEN_CONNECTIONS_H_
#define _OPEN_CONNECTIONS_H_

#include <memory> /* shared_ptr */
#include <mutex>
#include <set>
#include <vector>

class Fabric_connection;

namespace Component
{
  class IFabric_connection;
}

class Open_cnxns
{
public:
  using cnxn_t = std::shared_ptr<Fabric_connection>;
#if 0
  using open_t = std::map<Fabric_connection *, cnxn_t>;
#else
  using open_t = std::set<cnxn_t>;
#endif
private:
  std::mutex _m; /* protects _m */
  using guard = std::unique_lock<std::mutex>;
  std::set<cnxn_t> _s;
public:
  Open_cnxns();
  void add(cnxn_t c);
  void remove(Component::IFabric_connection *);
  std::vector<Component::IFabric_connection *> enumerate();
};

#endif
