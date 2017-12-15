/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#pragma once

#ifndef __COMPONENT_API_MEMORY_H_
#define __COMPONENT_API_MEMORY_H_

#include <component/base.h>

namespace Component {

typedef uint64_t io_buffer_t;

enum {
  NUMA_NODE_ANY = -1,
};

// API class - specifically minimize dependencies

/** 
 * Zero-copy memory management interface
 * 
 */
class IZerocopy_memory  : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0x7ac887cf,0xc0fe,0x4dc1,0xae7d,0x09,0x35,0xd7,0xd3,0xb2,0x81);
                         
   /** 
   * Allocate a contiguous memory region that can be used for IO
   * 
   * @param size Size of memory in bytes
   * @param alignment Alignment in bytes
   * @param numa_node NUMA node (-1) for any
   * 
   * @return Handle to IO memory region
   */
  virtual io_buffer_t allocate_io_buffer(size_t size, unsigned alignment, int numa_node) = 0;

  /** 
   * Re-allocate area of memory
   * 
   * @param io_mem Memory handle (from allocate_io_buffer)
   * @param size New size of memory in bytes
   * @param alignment Alignment in bytes
   * 
   * @return S_OK or E_NO_MEM (unable) or E_NOT_IMPL
   */
  virtual status_t realloc_io_buffer(io_buffer_t io_mem, size_t size, unsigned alignment) = 0;
  
  /** 
   * Free a previously allocated buffer
   * 
   * @param io_mem Handle to IO memory allocated by allocate_io_buffer
   * 
   * @return S_OK on success
   */
  virtual status_t free_io_buffer(io_buffer_t io_mem) = 0;

  /** 
   * Register memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual io_buffer_t register_memory_for_io(void * vaddr, addr_t paddr, size_t len) = 0;

  /** 
   * Unregister memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual void unregister_memory_for_io(void * vaddr, size_t len) = 0;

  /** 
   * Get pointer (virtual address) to start of IO buffer
   * 
   * @param buffer IO memory buffer handle
   * 
   * @return pointer
   */
  virtual void * virt_addr(io_buffer_t buffer) = 0;

  /** 
   * Get physical address of buffer
   * 
   * @param buffer IO memory buffer handle
   * 
   * @return physical address
   */
  virtual addr_t phys_addr(io_buffer_t buffer) = 0;

  /** 
   * Get size of memory buffer
   * 
   * @param buffer IO memory buffer handle
   * 
   * @return 
   */
  virtual size_t get_size(io_buffer_t buffer) = 0;
  
};

} /*< Component */

#endif
