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
#include <api/components.h>
namespace Component
{

class IBlock_allocator;

class IBlock_allocator_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfac38fab,0x35ab,0x4152,0xa1cc,0xeb,0x20,0xe6,0x9a,0xe0,0x79);
  
  /** 
   * Open allocator
   * 
   * @param pmem Persistent memory area to store bitmaps
   * @param max_lba Maximum LBA to track
   * @param name Allocator persistent identifier
   * 
   * @return 
   */
  virtual IBlock_allocator * open_allocator(Component::IPersistent_memory * pmem,
                                            size_t max_lba,
                                            persist_id_t persistent_id,
                                            int numa_node = NUMA_NODE_ANY,
                                            bool force_init = false){
    throw API_exception("not implemented.");
  }



/** 
   * Open allocator using AEP
   * 
   * @param max_lba Maximum LBA to track
   * @param name Allocator persistent identifier
   * 
   * @return 
   */
  virtual IBlock_allocator * open_allocator(size_t max_lba,
                                            persist_id_t persistent_id,
                                            int numa_node = NUMA_NODE_ANY,
                                            bool force_init = false){
    throw API_exception("not implemented.");
  }
};


/** 
 * General allocator interface.  Units are normally blocks or bytes.
 * 
 */
class IBlock_allocator : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xd7338fab,0x35ab,0x4152,0xa1cc,0xeb,0x20,0xe6,0x9a,0xe0,0x79);

  /** 
   * Allocate N contiguous blocks
   * 
   * @param size Number of blocks to allocate
   * @param handle If provided handle can be used to accelerate free
   * 
   * @return Logical block address of start of allocation.
   */
  virtual lba_t alloc(size_t size, void** handle = nullptr) = 0;

  /** 
   * Free a previous allocation
   * 
   * @param addr Logical block address of allocation
   * @param handle If provided handle can be used to accelerate free
   */
  virtual void free(lba_t addr, void* handle = nullptr) = 0;

  /** 
   * Attempt to resize an allocation without relocation
   * 
   * @param addr Logical block address of allocation
   * @param size New size in blocks
   * 
   * @return S_OK on successful resize; E_FAIL otherwise
   */
  virtual status_t resize(lba_t addr, size_t size) = 0;

  /** 
   * Left-trim an allocation
   * 
   * @param addr Address of contigous block
   * @param size_t Size to trimm from the left
   * 
   * @return New, trimmerd, logical block address of start
   */
  virtual lba_t ltrim(lba_t addr, size_t) { return E_NOT_IMPL; }
  
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

  virtual void dump_info() { };
};

} // namespace Component

#endif
