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

#include "fd_unblock_set_monitor.h"

fd_unblock_set_monitor:: fd_unblock_set_monitor(std::mutex &m_, std::set<int> &s_, int fd_)
  : _m(m_)
  , _s(s_)
  , _fd(fd_)
{
  guard g{_m};
  _s.insert(_fd);
}

fd_unblock_set_monitor::~fd_unblock_set_monitor()
{
  guard g{_m};
  _s.erase(_fd);
}
