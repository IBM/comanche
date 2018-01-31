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
  Author(s):
  Copyright (C) 2016, Daniel G. Waddington <daniel.waddington@ibm.com>
  Copyright (C) 2014, Daniel G. Waddington <daniel.waddington@acm.org>
*/

#ifndef __COMMON_CPU_UTILS_H__
#define __COMMON_CPU_UTILS_H__

#include <common/logging.h>
#include <common/types.h>
#include <pthread.h>
#include <sched.h>
#include <string>

#if defined(__cplusplus)

#if defined(__x86_64__) || defined(__x86_32__)
#define CACHE_LINE_SIZE 64  // Intel only
#define CACHE_LINE_SHIFT 6
#else
#error Unsupported HW architecture.
#endif

#if defined(__unix__)
class cpu_mask_t
{
 private:
  cpu_set_t cpu_set_;

 public:
  cpu_mask_t()
  {
    __builtin_memset(&cpu_set_, 0, sizeof(cpu_set_t));
  }

  cpu_mask_t(const cpu_mask_t& inst)
  {
    __builtin_memcpy(&cpu_set_, &inst, sizeof(cpu_set_t));
  }

  void add_core(int cpu)
  {
    CPU_SET(cpu, &cpu_set_);
  }

  bool check_core(int cpu)
  {
    return CPU_ISSET(cpu, &cpu_set_);
  }

  void set_mask(uint64_t mask)
  {
    int current = 0;
    while (mask > 0) {
      if (mask & 0x1ULL) {
        CPU_SET(current, &cpu_set_);
      }
      current++;
      mask = mask >> 1;
    }
  }

  void clear()
  {
    CPU_ZERO(&cpu_set_);
  }
  
  size_t size()
  {
    return sizeof(cpu_set_t);
  }

  bool is_something_set()
  {
    return (CPU_COUNT(&cpu_set_) > 0);
  }

  int count()
  {
    return CPU_COUNT(&cpu_set_);
  }
  
  void dump()
  {
    for (unsigned i = 0; i < 64; i++) {
      if (CPU_ISSET(i, &cpu_set_))
        printf("1");
      else
        printf("0");
    }
    printf("\n");
  }

  const cpu_set_t* cpu_set()
  {
    return &cpu_set_;
  }
};
#elif defined(__MACH__)

class cpu_mask_t
{
 private:
 public:
  cpu_mask_t()
  {
    PWRN("thread affinity not implemented for Mac OS");
  }

  cpu_mask_t(const cpu_mask_t& inst)
  {
    PWRN("thread affinity not implemented for Mac OS");
  }

  void set_bit(int cpu)
  {
    PWRN("thread affinity not implemented for Mac OS");
  }

  void set_mask(uint64_t mask)
  {
    PWRN("thread affinity not implemented for Mac OS");
  }

  size_t size()
  {
    PWRN("thread affinity not implemented for Mac OS");
    return 0;
  }

  bool is_set()
  {
    PWRN("thread affinity (is_set) not implemented for Mac OS");
    return false;
  }

  void dump()
  {
    PWRN("thread affinity not implemented for Mac OS");
  }
};

#else
#error Platform not supported.
#endif

int set_cpu_affinity_mask(cpu_mask_t& mask);
int set_cpu_affinity(unsigned long mask);
status_t string_to_mask(std::string def, cpu_mask_t& mask);

#endif

#endif
