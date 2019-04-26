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

#ifndef __BLOCK_ALLOC_COMPONENT_H__
#define __BLOCK_ALLOC_COMPONENT_H__

#include <common/memory.h>
#include <common/mpmc_bounded_queue.h>
#include <api/block_allocator_itf.h>

class Segment;

class Block_allocator : public Component::IBlock_allocator
{  
private:
  static constexpr bool option_DEBUG = true;

public:
  /** 
   * Constructor
   * 
   * @param block_device Block device interface
   * 
   */
  Block_allocator(Component::IPersistent_memory * pmem, size_t max_lba, Component::persist_id_t id, int numa_node, bool force_init);

  /** 
   * Destructor
   * 
   */
  virtual ~Block_allocator();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x7f23a368,0xd93b,0x488b,0x95cf,0x8c,0x77,0x3c,0xc5,0xaa,0xf3);
  
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

  Common::Std_allocator                 _stdalloc;
  Component::IPersistent_memory *       _pmem;
  Component::IPersistent_memory::pmem_t _pmem_pages;
  size_t                                _num_segments;
  unsigned                              _last_bin_index;
  Segment *                             _root_segment;
  
  std::vector<Common::Mpmc_bounded_lfq<Segment *>*> _vofsl;
};


class Block_allocator_factory : public Component::IBlock_allocator_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac3a368,0xd93b,0x488b,0x95cf,0x8c,0x77,0x3c,0xc5,0xaa,0xf3);
  
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
   * @param block_device Block device interface
   * 
   * @return Pointer to allocator instance. Ref count = 1. Release ref to delete.
   */
  virtual Component::IBlock_allocator * open_allocator(Component::IPersistent_memory * pmem,
                                                       size_t max_lba,
                                                       Component::persist_id_t id,
                                                       int numa_node,
                                                       bool force_init) override
  {
    if(pmem == nullptr)
      throw Constructor_exception("%s: bad persistent memory interface param", __PRETTY_FUNCTION__);
    
    Component::IBlock_allocator * obj = static_cast<Component::IBlock_allocator*>
      (new Block_allocator(pmem, max_lba, id, numa_node, force_init));
    
    obj->add_ref();
    return obj;
  }

virtual Component::IBlock_allocator * open_allocator(  size_t max_lba,
                                                       std::string path,
                                                       std::string name,
                                                       int numa_node,
                                                       bool force_init) override{
    throw API_exception("not implemented.");
  }

};



#endif
