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
#include <common/memory.h>

namespace Common {
/**
 * Allocate memory at a specific region.  Mainly for debugging purposes.
 *
 * @param size Size of region to allocate in bytes.
 * @param addr Location to allocate at.
 *
 * @return
 */
void* malloc_at(size_t size, addr_t addr) {
  static addr_t hint = 0xEE00000000ULL;

  if (addr == 0) {
    addr = hint;
    hint += 0x10000000ULL;
  }

  void* ptr =
      ::mmap((void*)(addr), size + sizeof(uint32_t), PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, 0, 0);

  *((uint32_t*)ptr) = size;
  return (void*)(((addr_t)ptr) + sizeof(uint32_t));
}

void free_at(void* ptr) {
  void* base = (void*)(((addr_t)ptr) - sizeof(uint32_t));
  ::munmap(base, *((uint32_t*)ptr));
}
}
