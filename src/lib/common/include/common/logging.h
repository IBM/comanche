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

#ifndef __COMMON_LOGGING_H__
#define __COMMON_LOGGING_H__

#include <assert.h>
#include <stdio.h>

#define NORMAL_CYAN "\033[36m"
#define NORMAL_MAGENTA "\033[35m"
#define NORMAL_BLUE "\033[34m"
#define NORMAL_YELLOW "\033[33m"
#define NORMAL_GREEN "\033[32m"
#define NORMAL_RED "\033[31m"

#define BRIGHT "\033[1m"
#define NORMAL_XDK "\033[0m"
#define RESET "\033[0m"

#define BRIGHT_CYAN "\033[1m\033[36m"
#define BRIGHT_MAGENTA "\033[1m\033[35m"
#define BRIGHT_BLUE "\033[1m\033[34m"
#define BRIGHT_YELLOW "\033[1m\033[33m"
#define BRIGHT_GREEN "\033[1m\033[32m"
#define BRIGHT_RED "\033[1m\033[31m"

#define WHITE_ON_RED "\033[41m"
#define WHITE_ON_GREEN "\033[42m"
#define WHITE_ON_YELLOW "\033[43m"
#define WHITE_ON_BLUE "\033[44m"
#define WHITE_ON_MAGENTA "\033[44m"

#define ESC_LOG NORMAL_GREEN
#define ESC_DBG NORMAL_YELLOW
#define ESC_INF NORMAL_CYAN
#define ESC_WRN NORMAL_RED
#define ESC_ERR BRIGHT_RED
#define ESC_END "\033[0m"

#undef PDBG
#undef PLOG
#undef PTEST
#undef PERR
#undef PWRN
#undef PASSERT
#undef PNOTICE
#undef PMAJOR
#undef TRACE

#ifdef CONFIG_DEBUG
#define PDBG(f, ...) fprintf(stderr, "%s[DBG]:%s: " f "%s\n", ESC_DBG, __FUNCTION__, ##__VA_ARGS__, ESC_END);
#define PLOG(f, ...) fprintf(stderr, "%s[LOG]:" f "%s\n", ESC_LOG, ##__VA_ARGS__, ESC_END)

#else  //--------------
#define PDBG(f, ...) \
  {                   \
  }
#define PLOG(f, ...) \
  {                   \
  }
#endif

#define PTEST(f, ...) fprintf(stdout, "[TEST]: %s:" f "\n", __FUNCTION__, ##__VA_ARGS__)

#define PINF(f, ...) fprintf(stderr, "%s" f "%s\n", ESC_INF, ##__VA_ARGS__, ESC_END)
#define PWRN(f, ...) fprintf(stderr, "%s[WRN]:" f "%s\n", ESC_WRN, ##__VA_ARGS__, ESC_END)
#define PERR(f, ...) fprintf(stderr, "%sERROR %s:" f "%s\n", ESC_ERR, __FUNCTION__, ##__VA_ARGS__, ESC_END);
#define PNOTICE(f, ...) \
  fprintf(stderr, "%sNOTICE %s:" f "%s\n", BRIGHT_RED, __FUNCTION__, ##__VA_ARGS__, ESC_END);
#define PMAJOR(f, ...) \
  fprintf(stdout, "%s[+] " f "%s\n", NORMAL_BLUE, ##__VA_ARGS__, ESC_END);
#define POK(f, ...) \
  fprintf(stderr, "%sOK %s:" f "%s\n", NORMAL_MAGENTA, __FUNCTION__, ##__VA_ARGS__, ESC_END);

#define PEXCEP(f, ...) fprintf(stderr, "%sException:" f "%s\n", ESC_ERR, ##__VA_ARGS__, ESC_END)

#ifdef CONFIG_DEBUG
#define PASSERT(cond, f, ...)                                                                    \
  if (!cond) {                                                                                    \
    fprintf(stderr, "%s[KIVATI]: ASSERT FAIL %s:" f "\n%s", ESC_ERR, __FUNCTION__, ##__VA_ARGS__, ESC_END); \
    assert(cond);                                                                                 \
  }
#else
#define PASSERT(cond, f, ...) \
  {                            \
  }
#endif

#define TRACE() fprintf(stderr, "[TRACE]: %s\n", __FUNCTION__)
#define THREAD_ROLE(ROLE) PLOG("thread (%p) role:%s", (void*)pthread_self(), ROLE)

#endif  // __COMMON_LOGGING_H__
