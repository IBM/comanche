/*
   Copyright [2017-2019] [IBM Corporation]
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

#include "physical_memory.h"

#include <rte_malloc.h>
#include <stdlib.h>
#include <sstream>
#include <string>

#include <api/memory_itf.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <core/stacktrace.h>

#include <cxxabi.h>    // for __cxa_demangle
#include <dlfcn.h>     // for dladdr
#include <execinfo.h>  // for backtrace
#include <stdio.h>

#include <sstream>
#include <string>

extern "C" {
#include <spdk/env.h>
}

bool DPDK::validate_memory(void* ptr, size_t* psize) {
  return rte_malloc_validate(ptr, psize) == -1 ? false : true;
}

namespace Core
{
/**
 * Allocate a contiguous memory region that can be used for IO
 *
 * @param size Size of memory in bytes
 * @param alignment Alignment
 * @param numa_node NUMA zone, or Component::NUMA_NODE_ANY
 *
 * @return Handle to IO memory region
 */
Component::io_buffer_t Physical_memory::allocate_io_buffer(size_t size,
                                                           unsigned alignment,
                                                           int numa_node) {
  void* ptr = rte_malloc_socket(NULL, size, alignment, numa_node);
  if (!ptr) {
    if (option_DEBUG) {
      rte_dump_physmem_layout(stderr);
      rte_malloc_dump_stats(stderr, NULL);
    }

    throw General_exception(
        "RTE out of memory (request size=%ld alignment=%u numa_node=%d)", size,
        alignment, numa_node);
  }

  if (option_DEBUG) {
    PLOG("##+ allocated DPDK memory: %ld @ %p", size, ptr);
    PINF("%s", Core::stack_trace().c_str());
  }

  return reinterpret_cast<Component::io_buffer_t>(ptr);
}

/**
 * Re-allocate area of memory
 *
 * @param io_mem Memory handle (from allocate_io_buffer)
 * @param size New size of memory in bytes
 * @param alignment Alignment in bytes
 *
 * @return S_OK or E_NO_MEM
 */
status_t Physical_memory::realloc_io_buffer(Component::io_buffer_t io_mem,
                                            size_t size, unsigned alignment) {
  void* ptr = reinterpret_cast<void*>(io_mem);
  ptr = rte_realloc(ptr, size, alignment);
  if (!ptr) return E_NO_MEM;
  return S_OK;
}

/**
 * Free a previously allocated buffer
 *
 * @param io_mem Handle to IO memory allocated by allocate_io_buffer
 *
 * @return S_OK on success
 */
status_t Physical_memory::free_io_buffer(Component::io_buffer_t io_mem) {
  void* ptr = reinterpret_cast<void*>(io_mem);
  if (ptr == nullptr) return E_INVAL;

  if (option_DEBUG) PLOG("##- freeing DPDK memory: %p", ptr);

  rte_free(ptr);
  return S_OK;
}

addr_t xms_get_phys(void* vaddr) {
  enum {
    IOCTL_CMD_GETBITMAP = 9,
    IOCTL_CMD_GETPHYS = 10,
  };

  typedef struct {
    addr_t vaddr;
    addr_t out_paddr;
  } __attribute__((packed)) IOCTL_GETPHYS_param;

  /* use xms to get physical memory address  */
  IOCTL_GETPHYS_param ioparam = {0};
  {
    int fd = open("/dev/xms", O_RDWR);

    ioparam.vaddr = (addr_t) vaddr;

    int rc = ioctl(fd, IOCTL_CMD_GETPHYS, &ioparam);  // ioctl call
    if (rc != 0) {
      PERR("%s(): ioctl failed on xms module: %s\n", __func__, strerror(errno));
    }
    close(fd);
  }
  return ioparam.out_paddr;
}

/**
 * Register memory for DMA with the SPDK subsystem.
 *
 * @param vaddr
 * @param len
 */

Component::io_buffer_t Physical_memory::register_memory_for_io(void* vaddr,
                                                               addr_t paddr,
                                                               size_t len) {
  if (!check_aligned(vaddr, MB(2)))
    throw API_exception(
        "register_memory_for_io requires vaddr be 2MB alignment");

  if (!check_aligned(paddr, MB(2)))
    throw API_exception(
        "register_memory_for_io requires paddr be 2MB alignment");

  int rc = spdk_mem_register(vaddr, len);

  if (spdk_vtophys(vaddr) != paddr)
    throw General_exception(
        "SPDK address registration check failed (rc=%d), spdk_vtophys returns "
        "physcial address 0x%lx != 0x%lx",
        rc, spdk_vtophys(vaddr), paddr);

  return reinterpret_cast<Component::io_buffer_t>(vaddr);
}

/**
 * Unregister memory for DMA with the SPDK subsystem.
 *
 * @param vaddr
 * @param len
 */
void Physical_memory::unregister_memory_for_io(void* vaddr, size_t len) {
  spdk_mem_unregister(vaddr, len);
}

/**
 * Get pointer (address) to start of IO buffer
 *
 * @param buffer IO buffer handle
 *
 * @return pointer
 */
void* Physical_memory::virt_addr(Component::io_buffer_t buffer) {
  return reinterpret_cast<void*>(buffer);
}

/**
 * Get physical address
 *
 * @param buffer
 *
 * @return
 */
addr_t Physical_memory::phys_addr(Component::io_buffer_t buffer) {
  return spdk_vtophys(reinterpret_cast<void*>(buffer));
}

/**
 * Get size of memory buffer
 *
 * @param buffer IO memory buffer handle
 *
 * @return Size in bytes of the IOB
 */
size_t Physical_memory::get_size(Component::io_buffer_t buffer) {
  void* p = reinterpret_cast<void*>(buffer);
  size_t s;
  if (rte_malloc_validate(p, &s) == -1)
    throw General_exception("rte_malloc_validate failed");

  return s;
}

}  // namespace Core
