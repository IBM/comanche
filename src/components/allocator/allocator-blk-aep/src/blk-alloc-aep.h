/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */


#ifndef __BLOCK_ALLOC_AEP_COMPONENT_H__
#define __BLOCK_ALLOC_AEP_COMPONENT_H__

#include <common/memory.h>
#include <common/mpmc_bounded_queue.h>

#include "bitmap-tx.h"

#include <api/block_allocator_itf.h>


class Block_allocator_AEP : public Component::IBlock_allocator
{  
private:
  static constexpr bool option_DEBUG = true;
  std::string _pool_name; // path to find the allocation info

public:
  /** 
   * Constructor
   * 
   * @param max_lba maximum blocks this allcator can support
   * @param id  the id for this specific allocator, this should be prefixed with the actual kvstore pool name
   *
   * this is a persistent block allocator, if the id already exist, it will try to reuse the mapping information
   * 
   */
  Block_allocator_AEP(size_t max_lba, std::string path, std::string name, int numa_node, bool force_init);

  /** 
   * Destructor
   * 
   */
  virtual ~Block_allocator_AEP();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x7f23a368,0x1993,0x488b,0x95cf,0x8c,0x77,0x3c,0xc5,0xaa,0xf3);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IBlock_allocator::iid()) {
      return (void *) static_cast<Component::IBlock_allocator*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

public:

  /* IBlock_allocator */

  /** 
   * Allocate N contiguous blocks
   * 
   * @param size Number of blocks to allocate
   * @param handle If provided handle can be used to accelerate free
   * 
   * @return Logical block address of start of allocation.
   */
  virtual lba_t alloc(size_t size, void** handle) override;

  /** 
   * Free a previous allocation
   * 
   * @param addr Logical block address of allocation
   * @param handle If provided handle can be used to accelerate free
   */
  virtual void free(lba_t addr, void* handle) override;

  /** 
   * Attempt to resize an allocation without relocation
   * For persist allocator, pass (0,0) to it will the objpool info from pmem
   * 
   * @param addr Logical block address of allocation
   * @param size New size in blocks
   * 
   * @return S_OK on successful resize; E_FAIL otherwise
   */
  virtual status_t resize(lba_t addr, size_t size) override;
  
  /** 
   * Get number of free units
   * 
   * 
   * @return Free capacity in units
   */
  virtual size_t get_free_capacity() override;

  /** 
   * Get total capacity
   * 
   * 
   * @return Capacity in units
   */
  virtual size_t get_capacity() override;

  virtual void dump_info() override;

private:

  // name to create a obj pool for metadata

  PMEMobjpool *_pop;
  TOID(struct bitmap_tx) _map;
  size_t _nbits; // length of bitmap
};


class Block_allocator_AEP_factory : public Component::IBlock_allocator_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac3a368,0x1993,0x488b,0x95cf,0x8c,0x77,0x3c,0xc5,0xaa,0xf3);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IBlock_allocator_factory::iid()) {
      return (void *) static_cast<Component::IBlock_allocator_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  /** 
   * Open an allocator
   * 
   * 
   * @return Pointer to allocator instance. Ref count = 1. Release ref to delete.
   */

virtual Component::IBlock_allocator * open_allocator(  size_t max_lba,
                                                       const std::string path,
                                                       const std::string name,
                                                       int numa_node,
                                                       bool force_init) override{
{
    Component::IBlock_allocator * obj = static_cast<Component::IBlock_allocator*>
      (new Block_allocator_AEP(max_lba, path, name, numa_node, force_init));
    
    obj->add_ref();
    return obj;
  }
  return NULL; 
  }

};

#endif
