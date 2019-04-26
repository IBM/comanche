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
#ifndef _TEST_PINGPONG_STAT_H_
#define _TEST_PINGPONG_STAT_H_

#include <chrono> /* high_resolution_clock */
#include <cstdint> /* uint64_t */

class pingpong_stat
{
  std::chrono::high_resolution_clock::time_point _start;
  std::chrono::high_resolution_clock::time_point _stop;
  std::uint64_t _poll_count;
public:
  pingpong_stat()
    : _start(std::chrono::high_resolution_clock::time_point::min())
    , _stop()
    , _poll_count(0U)
  {}
  void do_start() { _start = std::chrono::high_resolution_clock::now(); }
  void do_stop(std::uint64_t poll_count)
  {
    _stop = std::chrono::high_resolution_clock::now();
    _poll_count = poll_count;
  }
  std::chrono::high_resolution_clock::time_point start() const { return _start; }
  std::chrono::high_resolution_clock::time_point stop() const { return _stop; }
  std::uint64_t poll_count() const { return _poll_count; }
};

#endif
