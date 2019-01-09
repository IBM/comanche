/*
   Copyright [2017] [IBM Corporation]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#include "nvme_buffer.h"

#include <spdk/nvme.h>
extern "C" {
#include <rte_config.h>
#include <rte_malloc.h>
#include <rte_mempool.h>
}

addr_t Nvme_buffer::get_physical(void* io_buffer) {
  return rte_malloc_virt2iova(io_buffer);
}

void* Nvme_buffer::allocate_io_buffer(size_t size_to_allocate,
                                      unsigned numa_socket, bool zero_init) {
  assert(size_to_allocate % 64 == 0);

  void* ptr;

  if (zero_init) {
    ptr = rte_zmalloc_socket(NULL, size_to_allocate, 4096 /*alignment*/,
                             numa_socket);
  } else {
    ptr = rte_malloc_socket(NULL, size_to_allocate, 4096 /*alignment*/,
                            numa_socket);
  }

  //  PLOG("allocated Nvme_buffer @ phys:%lx", rte_malloc_virt2phy(ptr));

  if (!ptr)
    throw new Constructor_exception(
        "rte_zmalloc failed in Nvme_buffer::allocate_io_buffer");

  return ptr;
}

void Nvme_buffer::free_io_buffer(void* buffer) {
  assert(buffer);
  rte_free(buffer);
}
