#ifndef __ZEROCOPY_PASSTHROUGHH__
#define __ZEROCOPY_PASSTHROUGHH__

#include <api/memory_itf.h>
#include <api/block_itf.h>

namespace comanche
{

template <class __Base>
class Zerocopy_passthrough_impl : public __Base
{
public:

  Zerocopy_passthrough_impl() : _lower_layer(nullptr) {
  }

  virtual ~Zerocopy_passthrough_impl() {
    if(_lower_layer)
      _lower_layer->release_ref();
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
  virtual Component::io_buffer_t allocate_io_buffer(size_t size,
                                                    unsigned alignment,
                                                    int numa_node) override {
    assert(_lower_layer);
    return _lower_layer->allocate_io_buffer(size,alignment,numa_node);
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
  virtual status_t realloc_io_buffer(Component::io_buffer_t io_mem, size_t size, unsigned alignment) override {
    assert(_lower_layer);
    return _lower_layer->realloc_io_buffer(io_mem,size,alignment);
  }

  /** 
   * Free a previously allocated buffer
   * 
   * @param io_mem Handle to IO memory allocated by allocate_io_buffer
   * 
   * @return S_OK on success
   */
  virtual status_t free_io_buffer(Component::io_buffer_t io_mem) override {
    assert(_lower_layer);
    return _lower_layer->free_io_buffer(io_mem);
  }

  /** 
   * Register memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual Component::io_buffer_t register_memory_for_io(void * vaddr, addr_t paddr, size_t len) override {
    assert(_lower_layer);
    return _lower_layer->register_memory_for_io(vaddr, paddr, len);
  }

  /** 
   * Unregister memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual void unregister_memory_for_io(void *vaddr, size_t len) override {
    assert(_lower_layer);
    _lower_layer->unregister_memory_for_io(vaddr,len);
  }
  
  /** 
   * Get pointer (virtual address) to start of IO buffer
   * 
   * @param buffer IO buffer handle
   * 
   * @return pointer
   */
  virtual void * virt_addr(Component::io_buffer_t buffer) override {
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
  virtual addr_t phys_addr(Component::io_buffer_t buffer) override {
    return _lower_layer->phys_addr(buffer);
  }
  
  /** 
   * Get size of memory buffer
   * 
   * @param buffer IO memory buffer handle
   * 
   * @return 
   */
  virtual size_t get_size(Component::io_buffer_t buffer) override {
    return _lower_layer->get_size(buffer);
  }
  
protected:
  Component::IBlock_device * _lower_layer;
  
};

}


#endif //__ZEROCOPY_PASSTHROUGHH__
