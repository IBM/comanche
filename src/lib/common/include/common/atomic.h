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
  Copyright (C) 2011-2016, Daniel G. Waddington <daniel.waddington@acm.org>

*/

#ifndef __COMMON_ATOMIC_H__
#define __COMMON_ATOMIC_H__

#include <common/types.h>
#include <common/utils.h>

#if defined(__x86_64__)
#if defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_8)
#define USE_GCC_CAS_INTRINSIC
#else
#error 64-bit target architecture must support atomic instrinsics.
#endif
#endif

#if defined(__i386__) || defined(__i686__) || defined(__i586__)
#if defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4)
#define USE_GCC_CAS_INTRINSIC
#else
inline bool __sync_bool_compare_and_swap(volatile int *dest, int cmp_val,
                                         int new_val) {
  int tmp;
  __asm__ __volatile__("lock cmpxchgl %1, %3 \n\t"
                       : "=a"(tmp)     /* 0 EAX, return val */
                       : "r"(new_val), /* 1 reg, new value */
                         "0"(cmp_val), /* 2 EAX, compare value */
                         "m"(*dest)    /* 3 mem, destination operand */
                       : "memory", "cc");

  return tmp == cmp_val;
}
#endif
#endif

#if defined(__arm__)
#if defined(__GCC_HAVE_SYNC_COMPARE_AND_SWAP_4)
#define USE_GCC_CAS_INTRINSIC
#else
#error 32-bit target architecture must support atomic instrinsics.
#endif
#endif

namespace Common
{
#if defined(USE_GCC_CAS_INTRINSIC) && defined(__cplusplus)

/**
 * Class for atomics
 *
 */
class Atomic {
 private:
  volatile atomic_t __atomic __attribute__((aligned(sizeof(atomic_t))));

 public:
  Atomic() : __atomic(0) {}
  Atomic(atomic_t v) : __atomic(v) {}
  virtual ~Atomic() {}

  inline void init() { __atomic = (atomic_t) 0; }
  inline void init_value(atomic_t a) { __atomic = a; }
  inline atomic_t get_value_unsafe() { return __atomic; }

  inline atomic_t read() { return __sync_fetch_and_add(&__atomic, 0UL); }

  /**
   * add - add value and return OLD value
   *
   */
  inline atomic_t add(Atomic v) {
    return __sync_fetch_and_add(&__atomic, v.get_value_unsafe());
  }

  /**
   * increment
   *
   */
  inline void increment() { __sync_fetch_and_add(&__atomic, 1UL); }

  /**
   * decrement - decrement
   *
   */
  inline void decrement() { __sync_fetch_and_sub(&__atomic, 1UL); }

  /**
   * increment_and_fetch - increment and return the
   * NEW value
   *
   */
  inline atomic_t increment_and_fetch() {
    return (__sync_fetch_and_add(&__atomic, 1UL) + 1);
  }

  /**
   * fetch_and_increment - increment and return the
   * OLD value
   *
   */
  inline atomic_t fetch_and_increment() {
    return (__sync_fetch_and_add(&__atomic, 1UL));
  }

  /**
   * decrement_and_fetch - increment and return the
   * NEW value
   *
   */
  inline atomic_t decrement_and_fetch() {
    return (__sync_fetch_and_sub(&__atomic, 1UL) - 1);
  }

  /**
   * fetch_and_decrement - increment and return the
   * OLD value
   *
   */
  inline atomic_t fetch_and_decrement() {
    return (__sync_fetch_and_sub(&__atomic, 1UL));
  }

  /**
   * compare_and_swap: three versions
   *
   */
  inline atomic_t compare_and_swap(Atomic oldval, Atomic newval) {
    return __sync_val_compare_and_swap(&__atomic, oldval.get_value_unsafe(),
                                       newval.get_value_unsafe());
  }
  inline atomic_t compare_and_swap(atomic_t oldval, atomic_t newval) {
    return __sync_val_compare_and_swap(&__atomic, oldval, newval);
  }

  static inline atomic_t compare_and_swap(atomic_t *mem, atomic_t oldval,
                                          atomic_t newval) {
    return __sync_val_compare_and_swap(mem, oldval, newval);
  }

  /**
   * Compare and swap; boolean version
   *
   * @return - true if condition was true and swap made
   */
  inline bool cas_bool(Atomic v, Atomic q) {
    return __sync_bool_compare_and_swap(&__atomic, v.get_value_unsafe(),
                                        q.get_value_unsafe());
  }
  static inline bool cas_bool(atomic_t *mem, atomic_t v, atomic_t q) {
    return __sync_bool_compare_and_swap(mem, v, q);
  }

  static inline void increment(atomic_t *mem) { __sync_fetch_and_add(mem, 1); }

  static inline void decrement(atomic_t *mem) { __sync_fetch_and_sub(mem, 1); }

} __attribute__((packed));

#else
#error Atomic requires sync intrinsics
#endif
}  // namespace Common

#endif
