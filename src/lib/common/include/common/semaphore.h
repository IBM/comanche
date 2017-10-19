/*
   eXokernel Development Kit (XDK)

   Samsung Research America Copyright (C) 2013

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.
*/


/*
  Authors:
  Copyright (C) 2017, Daniel G. Waddington <daniel.waddington@ibm.com>
*/

#ifndef __COMMON_SEMAPHORE_H__
#define __COMMON_SEMAPHORE_H__

#include <mutex>
#include <condition_variable>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>

namespace Common
{

class Semaphore {
public:
  Semaphore (int count_ = 0)
    : count(count_) {}

  inline void post()
  {
    std::unique_lock<std::mutex> lock(mtx);
    count++;
    cv.notify_one();
  }

  inline void wait()
  {
    std::unique_lock<std::mutex> lock(mtx);

    while(count == 0){
      cv.wait(lock);
    }
    count--;
  }

  inline bool wait_for(unsigned rel_ms)
  {
    std::unique_lock<std::mutex> lock(mtx);

    while(count == 0){
      if(cv.wait_for(lock, std::chrono::milliseconds(rel_ms))
         == std::cv_status::timeout) {
        return false;
      }
    }
    count--;
    return true;
  }

private:
  std::mutex mtx;
  std::condition_variable cv;
  int count;
};

} // namespace Common
#endif
