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


/*
 * Authors:
 *
 */

#ifndef _FABRIC_FD_UNBLOCK_SET_MONITOR_H_
#define _FABRIC_FD_UNBLOCK_SET_MONITOR_H_

#include <mutex>
#include <set>

class fd_unblock_set_monitor
{
  std::mutex &_m;
  std::set<int> &_s;
  int _fd;
public:
  explicit fd_unblock_set_monitor(std::mutex &m_, std::set<int> &s_, int fd_);
  ~fd_unblock_set_monitor();
};

#endif
