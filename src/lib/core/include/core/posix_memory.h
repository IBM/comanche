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

#ifndef __BLOCK_POSIX_MEMORY_H__
#define __BLOCK_POSIX_MEMORY_H__

#include <stdlib.h>

template <class __Base>
class Posix_memory : public __Base
{
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
  virtual Component::io_buffer_t allocate_io_buffer(size_t size, unsigned alignment, int /* numa_node */)
  {
    /* TO FIX: use DPDK - this memory is not pinned or physical contiguous !!!! */
    void* ptr;
    // ptr = aligned_alloc(alignment, size);
    // memset(ptr,0,size);
    ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS | MAP_LOCKED | MAP_POPULATE, 0, 0);


    //    int rc = posix_memalign(&ptr, alignment, size);
    if (ptr == nullptr)
      throw API_exception("Posix_memory: posix_memalign failed (alignment=%ld, size=%ld)", alignment, size);
    PLOG("iobuffer: allocated at %p", ptr);
    assert(ptr);
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
  virtual status_t realloc_io_buffer(Component::io_buffer_t io_mem, size_t size, unsigned alignment)
  {
    void* ptr = reinterpret_cast<void*>(io_mem);
    if (ptr == nullptr) throw API_exception("realloc_io_buffer: bad parameter");

    void* newptr = ::realloc(ptr, size);
    if (newptr == nullptr) return E_NO_MEM;
    if (newptr != ptr) throw Logic_exception("ptr changed from realloc");

    return S_OK;
  }


  /** 
   * Free a previously allocated buffer
   * 
   * @param io_mem Handle to IO memory allocated by allocate_io_buffer
   * 
   * @return S_OK on success
   */
  virtual status_t free_io_buffer(Component::io_buffer_t io_mem)
  {
    void* ptr = reinterpret_cast<void*>(io_mem);
    if (ptr == nullptr) throw API_exception("free_io_buffer: bad parameter");

    //::free(ptr); // TOFIX
  }

  /** 
   * Register memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual Component::io_buffer_t register_memory_for_io(void* vaddr, size_t len)
  {
  }

  /** 
   * Unregister memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual void unregister_memory_for_io(Component::io_buffer_t buffer)
  {
  }

  /** 
   * Get pointer (virtual address) to start of IO buffer
   * 
   * @param buffer IO buffer handle
   * 
   * @return pointer
   */
  virtual void* virt_addr(Component::io_buffer_t buffer)
  {
    return reinterpret_cast<void*>(buffer);
  }
};


#endif
