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
  Copyright (C) 2016, Daniel G. Waddington <daniel.waddington@ibm.com>
  Copyright (C) 2013, Daniel G. Waddington <d.waddington@samsung.com>
  Copyright (C) 2014, Daniel G. Waddington <daniel.waddington@acm.org>
*/

#ifndef __COMMON_ASSERT_H__
#define __COMMON_ASSERT_H__

#include <stdio.h>
#include "logging.h"

#if defined(__cplusplus)
extern "C"
#endif
    void
    panic(const char *format, ...) __attribute__((format(printf, 1, 2)));

#ifdef CONFIG_DEBUG
#if defined(__i386__) || defined(__x86_64__)
#define soft_stop(X)                                                           \
  ::printf("%s[%s:%d] STOP: %s%s\n", ESC_ERR, __FILE__, __LINE__, X, ESC_END); \
  asm("int3")
#else
#define soft_stop(X)                                                           \
  ::printf("%s[%s:%d] STOP: %s%s\n", ESC_ERR, __FILE__, __LINE__, X, ESC_END); \
  __builtin_trap()
#endif
#else
#define soft_stop(X)
#endif

#ifdef CONFIG_DEBUG
#define check_ok(X) assert(X == 0);
#else
#define check_ok(X) X
#endif

#ifdef CONFIG_DEBUG
#define assert_aligned(X, ALIGNMENT) \
  assert(!(((unsigned long) X) & (ALIGNMENT - 1UL)))
#else
#define assert_aligned(X, ALIGNMENT)
#endif

#ifdef CONFIG_DEBUG
#define ASSERT(X) (assert(X))
#else
#define ASSERT(X) (X)
#endif

#ifdef CONFIG_DEBUG
#define ASSERT(X) (assert(X))
#else
#define ASSERT(X) (X)
#endif

#ifdef CONFIG_DEBUG
bool check_ptr_valid(void *ptr, size_t len);
#define CHECK_PTR_VALID(PTR, LEN) assert(check_ptr_valid(PTR, LEN))
#else
#define CHECK_PTR_VALID(PTR, LEN)
#endif

#endif
