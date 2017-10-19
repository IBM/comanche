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

#ifndef __PHYSICAL_MEMORY_H__
#define __PHYSICAL_MEMORY_H__

#include <api/memory_itf.h>

namespace DPDK
{
bool validate_memory(void * ptr, size_t * psize = nullptr);
}

namespace Core
{

class Physical_memory
{
private:
  static constexpr bool option_DEBUG = false;
  
public:

  /** 
   * Allocate a contiguous memory region that can be used for IO
   * 
   * @param size Size of memory in bytes
   * @param alignment Alignment
   * @param numa_node NUMA zone, or Component::NUMA_NODE_ANY
   * 
   * @return Handle to IO memory region
   */
  virtual Component::io_buffer_t allocate_io_buffer(size_t size, unsigned alignment, int numa_node);
  
  /** 
   * Re-allocate area of memory
   * 
   * @param io_mem Memory handle (from allocate_io_buffer)
   * @param size New size of memory in bytes
   * @param alignment Alignment in bytes
   * 
   * @return S_OK or E_NO_MEM
   */
  virtual status_t realloc_io_buffer(Component::io_buffer_t io_mem, size_t size, unsigned alignment);

  /** 
   * Free a previously allocated buffer
   * 
   * @param io_mem Handle to IO memory allocated by allocate_io_buffer
   * 
   * @return S_OK on success
   */
  virtual status_t free_io_buffer(Component::io_buffer_t io_mem);

  /** 
   * Register memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual void register_memory_for_io(void* vaddr, size_t len);

  /** 
   * Unregister memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual void unregister_memory_for_io(void* vaddr, size_t len);

  /** 
   * Get pointer (virtual address) to start of IO buffer
   * 
   * @param buffer IO buffer handle
   * 
   * @return pointer
   */
  virtual void* virt_addr(Component::io_buffer_t buffer);

  /** 
   * Get physical address
   * 
   * @param buffer 
   * 
   * @return 
   */
  virtual addr_t phys_addr(Component::io_buffer_t buffer);

  /** 
   * Get size of memory buffer
   * 
   * @param buffer IO memory buffer handle
   * 
   * @return 
   */
  virtual size_t get_size(Component::io_buffer_t buffer);
  
};

} // namespace Core


/** 
 * Used to inline forward to the base class so that we can
 * avoid using templates and inherit the memory management directly.
 * 
 */

#define INLINE_FORWARDING_MEMORY_METHODS  \
  inline virtual Component::io_buffer_t                                 \
  allocate_io_buffer(size_t size, unsigned alignment, int numa_node) {  \
    return Physical_memory::allocate_io_buffer(size,alignment,numa_node); \
  }                                                                     \
                                                                        \
  inline virtual status_t realloc_io_buffer(Component::io_buffer_t io_mem, size_t size, \
                                    unsigned alignment) {               \
    return Physical_memory::realloc_io_buffer(io_mem, size, alignment); \
  }                                                                     \
                                                                        \
  inline virtual status_t free_io_buffer(Component::io_buffer_t io_mem) { \
    return Physical_memory::free_io_buffer(io_mem);                     \
  }                                                                     \
                                                                        \
  inline virtual void register_memory_for_io(void * vaddr, size_t len) { \
    Physical_memory::register_memory_for_io(vaddr, len);                \
  }                                                                     \
                                                                        \
  inline virtual void unregister_memory_for_io(void * vaddr, size_t len) { \
    Physical_memory::unregister_memory_for_io(vaddr,len);               \
  }                                                                     \
                                                                        \
  inline virtual void * virt_addr(Component::io_buffer_t buffer) {      \
    return Physical_memory::virt_addr(buffer);                          \
  }                                                                     \
                                                                        \
  inline virtual addr_t phys_addr(Component::io_buffer_t buffer) {      \
    return Physical_memory::phys_addr(buffer);                          \
  }                                                                     \
                                                                        \
  inline virtual size_t get_size(Component::io_buffer_t buffer) {      \
    return Physical_memory::get_size(buffer);                          \
  }



#endif
