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

#ifndef __BLOCKDEVICE_PASSTHROUGHH__
#define __BLOCKDEVICE_PASSTHROUGHH__

#include <api/memory_itf.h>
#include <api/block_itf.h>

namespace comanche
{
template <class __Base>
class Block_device_passthrough_impl : public __Base
{
 public:
  Block_device_passthrough_impl() : _lower_layer(nullptr)
  {
  }

  virtual ~Block_device_passthrough_impl()
  {
    if (_lower_layer) _lower_layer->release_ref();
  }

  /*------------*/
  /* IZero_copy */
  /*------------*/

  /** 
   * Allocate a contiguous memory region that can be used for IO
   * 
   * @param size Size of memory in bytes
   * @param alignment Alignment
   * 
   * @return Handle to IO memory region
   */
  virtual Component::io_buffer_t allocate_io_buffer(size_t size, unsigned alignment, int numa_node) override
  {
    assert(_lower_layer);
    return _lower_layer->allocate_io_buffer(size, alignment, numa_node);
  }

  /** 
   * Re-allocate area of memory
   * 
   * @param io_mem Memory handle (from allocate_io_buffer)
   * @param size New size of memory in bytes
   * @param alignment Alignment in bytes
   * 
   * @return S_OK or E_NO_MEM (unable) or E_NOT_IMPL
   */
  virtual status_t realloc_io_buffer(Component::io_buffer_t io_mem, size_t size, unsigned alignment) override
  {
    assert(_lower_layer);
    return _lower_layer->realloc_io_buffer(io_mem, size, alignment);
  }

  /** 
   * Free a previously allocated buffer
   * 
   * @param io_mem Handle to IO memory allocated by allocate_io_buffer
   * 
   * @return S_OK on success
   */
  virtual status_t free_io_buffer(Component::io_buffer_t io_mem) override
  {
    assert(_lower_layer);
    return _lower_layer->free_io_buffer(io_mem);
  }

  /** 
   * Register memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual Component::io_buffer_t register_memory_for_io(void* vaddr, size_t len) override
  {
    assert(_lower_layer);
    return _lower_layer->register_memory_for_io(vaddr, len);
  }

  /** 
   * Unregister memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual void unregister_memory_for_io(Component::io_buffer_t buffer) override
  {
    assert(_lower_layer);
    _lower_layer->unregister_memory_for_io(buffer);
  }

  /** 
   * Get pointer (virtual address) to start of IO buffer
   * 
   * @param buffer IO buffer handle
   * 
   * @return pointer
   */
  virtual void* virt_addr(Component::io_buffer_t buffer) override
  {
    assert(_lower_layer);
    return _lower_layer->virt_addr(buffer);
  }

  /** 
   * Get physical address of buffer
   * 
   * @param buffer IO buffer handle
   * 
   * @return physical address
   */
  virtual addr_t phys_addr(Component::io_buffer_t buffer) override
  {
    return _lower_layer->phys_addr(buffer);
  }

 protected:
  Component::IBlock_device* _lower_layer;
};
}


#endif  //__BLOCKDEVICE_PASSTHROUGHH__
