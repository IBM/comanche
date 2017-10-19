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
  Copyright (C) 2014, Daniel G. Waddington <daniel.waddington@acm.org>
*/

#include "common/dump_utils.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

void hexdump(void* data, unsigned len) {
  printf("HEXDUMP----------------------------------------------");
  assert(len > 0);
  uint8_t* d = (uint8_t*)data;
  for (unsigned i = 0; i < len; i++) {
    if (i % 16 == 0) {
      printf("\n0x%x:\t", i);
    }
    printf("%x%x ", 0xf & (d[i] >> 4), 0xf & d[i]);
  }
  printf("\n");
}

void asciidump(void* data, unsigned len) {
  printf("ASCIIDUMP----------------------------------------------");
  assert(len > 0);
  uint8_t* d = (uint8_t*)data;
  for (unsigned i = 0; i < len; i++) {
    if (i % 16 == 0) {
      printf("\n0x%x:\t", i);
    }
    printf("%c%c ", 0xf & (d[i] >> 4), 0xf & d[i]);
  }
  printf("\n");
}
