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

#include "open_cnxns.h"

#include <algorithm> /* lower_bound, transform */
#include <iterator> /* back_inserter */

using guard = std::unique_lock<std::mutex>;

Open_cnxns::Open_cnxns()
  : _m{}
  , _s{}
{}

void Open_cnxns::add(cnxn_t c_)
{
  guard g{_m};
  _s.insert(c_);
}

void Open_cnxns::remove(Fabric_memory_control *c_)
{
  guard g{_m};
  auto it =
    std::lower_bound(
      _s.begin()
      , _s.end()
      , c_
      , [] (const cnxn_t &e, Fabric_memory_control *c) { return e.get() < c; }
    );
  if ( it != _s.end() && it->get() == c_ )
  {
    _s.erase(it);
  }
}

std::vector<Fabric_memory_control *> Open_cnxns::enumerate()
{
  guard g{_m};
  std::vector<Fabric_memory_control *> v;

  std::transform(
    _s.begin()
    , _s.end()
    , std::back_inserter(v)
    , [] (const open_t::value_type &v_)
      {
        return &*v_;
      }
  );

  return v;
}
