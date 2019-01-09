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

#ifndef __COMMON_CHECKSUM_H__
#define __COMMON_CHECKSUM_H__

#include <common/types.h>
#include <zlib.h>

namespace Common
{
/**
 * Simple 32bit checksum
 *
 * @param buffer Memory area to checksum over
 * @param len Length of memory in bytes
 *
 * @return 32-bit checksum
 */
inline uint32_t chksum32(void *buffer, size_t len) {
  uint32_t chksum = crc32(0L, NULL, 0);
  chksum = crc32(chksum, (unsigned char *) buffer, len);
  return chksum;
}
}  // namespace Common

#endif
