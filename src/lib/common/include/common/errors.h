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
*/

#ifndef __ERRORS_H__
#define __ERRORS_H__

#ifndef ERROR_ENUMS
#define ERROR_ENUMS                                                     \
  enum {                                                                \
    S_OK = 0,                                                           \
    E_FAIL = -1,                                                        \
    E_INVALID_REQUEST = -2,                                             \
    E_INVAL = -2,                                                       \
    E_INSUFFICIENT_QUOTA = -3,                                          \
    E_NOT_FOUND = -4,                                                   \
    E_INSUFFICIENT_RESOURCES = -5,                                      \
    E_NO_RESOURCES = -6,                                                \
    E_INSUFFICIENT_SPACE = -7,                                          \
    E_INSUFFICIENT_BUFFER = -7,                                         \
    E_BUSY = -9,                                                        \
    E_TAKEN = -10,                                                      \
    E_LENGTH_EXCEEDED = -11,                                            \
    E_BAD_OFFSET = -12,                                                 \
    E_BAD_PARAM = -13,                                                  \
    E_NO_MEM = -14,                                                     \
    E_NOT_SUPPORTED = -15,                                              \
    E_OUT_OF_BOUNDS = -16,                                              \
    E_NOT_INITIALIZED = -17,                                            \
    E_NOT_IMPL = -18,                                                   \
    E_NOT_ENABLED = -19,                                                \
    E_SEND_TIMEOUT = -20,                                               \
    E_RECV_TIMEOUT = -21,                                               \
    E_BAD_FILE = -22,                                                   \
    E_FULL = -23,                                                       \
    E_EMPTY = -24,                                                      \
    E_INVALID_ARG = -25,                                                \
    E_BAD_SEMANTICS = -26,                                              \
    E_EOF = -27,                                                        \
    E_ALREADY = -28,                                                    \
    E_ALREADY_EXISTS = -28,                                             \
    E_NO_RESPONSE = -29,                                                \
    E_TIMEOUT = -30,                                                    \
    E_MAX_REACHED = -31,                                                \
    E_ERROR_BASE = -50,                                                 \
  }
#endif

ERROR_ENUMS; /* add to global namespace also */

#endif
