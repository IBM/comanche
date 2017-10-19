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
#ifndef __COMMON_STRUTILS_H__
#define __COMMON_STRUTILS_H__

#include <string>

namespace Common {
/**
 * Simple global substring replacement
 *
 * @param subject String to operate on
 * @param search String to search
 * @param replace Replacement string
 *
 * @return Modified string
 */
std::string string_replace(std::string& subject, const std::string& search,
                           const std::string& replace);

/**
 * Generate a random string from alpha-numeric characters
 *
 * @param length Length of string to generate
 *
 * @return Random string
 */
std::string random_string(size_t length);
}

#endif  // __COMMON_STRUTILS_H__
