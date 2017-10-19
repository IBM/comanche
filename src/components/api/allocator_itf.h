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

#ifndef __API_ALLOCATOR_ITF__
#define __API_ALLOCATOR_ITF__

#include <string>
#include <api/pmem_itf.h>

namespace Component
{

class IAllocator;

class IAllocator_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfac80d2f,0x04f2,0x439e,0xbc52,0xb1,0x3b,0x1c,0xdf,0x3e,0xc5);

  /** 
   * Open an allocator
   * 
   * @param pmem Persistent memory interface to store the metadata
   * @param id Identifier
   * @param capacity Number of unit elements
   * 
   * @return Pointer to allocator instance. Ref count = 1. Release ref to delete.
   */
  virtual IAllocator * open_allocator(Component::IPersistent_memory * pmem,
                                      std::string id,
                                      size_t n_elements) = 0;

};

/** 
 * General allocator interface.  Units are normally blocks or bytes.
 * 
 */
class IAllocator : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xe4a80d2f,0x04f2,0x439e,0xbc52,0xb1,0x3b,0x1c,0xdf,0x3e,0xc5);

  /** 
   * Allocate N contiguous units.
   * 
   * @param size Number of units to allocate
   * 
   * @return Address of start of allocation.
   */
  virtual addr_t alloc(size_t size) = 0;

  /** 
   * Free a previous allocation
   * 
   * @param addr Address of allocation
   * @param len Nunber of units to free.
   */
  virtual void free(addr_t addr, size_t len) = 0;

  /** 
   * Get number of free units
   * 
   * 
   * @return Free capacity in units
   */
  virtual size_t get_free_capacity() = 0;

  /** 
   * Get total capacity
   * 
   * 
   * @return Capacity in units
   */
  virtual size_t get_capacity() = 0;

  
};

} // namespace Component

#endif
