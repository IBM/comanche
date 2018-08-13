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

#ifndef _FD_PAIR_H_
#define _FD_PAIR_H_

#include <cstddef> /* size_t */

class Fd_pair
{
  int _pair[2];
public:
  /*
   * @throw std::system_error - creating fd pair
   */
  Fd_pair();
  explicit Fd_pair(int read_flags);
  int fd_read() const { return _pair[0]; }
  int fd_write() const { return _pair[1]; }
  /*
   * @throw std::system_error - reading fd of pair
   */
  std::size_t read(void *, std::size_t) const;
  /*
   * @throw std::system_error - writing fd of pair
   */
  std::size_t write(const void *, std::size_t) const;
  ~Fd_pair();
};

#endif
