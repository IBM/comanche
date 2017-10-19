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

#ifndef __SIMPLE_PAGER_H__
#define __SIMPLE_PAGER_H__

#include <list>

#include <common/logging.h>
#include <common/utils.h>
#include <api/pager_itf.h>
#include <api/region_itf.h>
#include "core/block_device_passthrough.h"

using namespace Component;

class Range_tracker;

class Simple_pager_component : public Component::IPager
{

private:
  static constexpr bool option_DEBUG = false;

public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xf53fd819,0xe157,0x4e69,0x9157,0xe8,0x49,0x51,0x77,0x07,0x17);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IPager::iid()) {
      return (void *) static_cast<Component::IPager *>(this);
    }
    else 
      return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }
  
public:
  /** 
   * Constructor
   * 
   * 
   */
  Simple_pager_component(size_t nr_pages,
                         std::string heap_set_id,
                         Component::IBlock_device * block_device,
                         bool force_init);


  /** 
   * Destructor
   * 
   */
  virtual ~Simple_pager_component();


  /* IPager interface */
  

  /** 
   * Get region (create or reuse)
   * 
   * @param id 
   * @param size 
   * @param reused
   * 
   * @return 
   */
  virtual void * get_region(std::string id,
                            size_t size,
                            bool& reused) override;
  
  /** 
   * 
   * Request page mapping/eviction pair
   *
   * @param virt_addr_faulted virtual address which trigger the faults
   * @param virt_addr_pinned, virt addr to access the pinned physical page(since the faulting address is no mapped yet)
   * @param p_phys_addr_faulted physical addr to map to faulted address
   * @param p_virt_addr_evicted 
   * @param is_young young means has been access before(swapped out previously)
   * 
   *
   */
  virtual void request_page(addr_t virt_addr_faulted,
                            addr_t *p_phys_addr_faulted,
                            addr_t *p_virt_addr_evicted) override;

  /** 
   * Clear mapping for a given range
   * 
   * @param vaddr Virtual address
   * @param size Size in bytes
   */
  virtual void clear_mappings(addr_t vaddr, size_t size) override;

  /** 
   * Flush data to storage
   * 
   * @param vaddr 
   * @param size 
   */
  virtual void flush(addr_t vaddr, size_t size) override;

  /** 
   * Get the write-atomicity size for the underlying device.
   * 
   * 
   * @return Atomicity size in bytes (usually 4096)
   */
  virtual size_t get_atomicity_size() override {
    VOLUME_INFO vi;
    _block_dev->get_volume_info(vi);    
    assert(vi.block_size > 0);
    return vi.block_size;
  }

private:
  void init_memory(size_t nr_pages);
  
private:
  struct page_descriptor {
    uint64_t gwid;
    addr_t vaddr;
    addr_t paddr;
  };

  IBlock_device *        _block_dev;
  IRegion_manager *      _rm;
  Range_tracker *        _tracker;
  struct page_descriptor * _pages;
  uint64_t               _request_num = 0;
  bool                   _running = false;
  size_t                 _nr_pages = 0;
  Component::io_buffer_t _iob;
  std::string            _heap_set_id;
  addr_t                 _phys_base;

};


class Simple_pager_component_factory : public Component::IPager_factory
{
public:
  DECLARE_COMPONENT_UUID(0xfacfd819,0xe157,0x4e69,0x9157,0xe8,0x49,0x51,0x77,0x07,0x17);


  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IPager_factory::iid()) {
      return (void *) static_cast<Component::IPager_factory *>(this);
    }
    else 
      return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  virtual Component::IPager * create(size_t nr_pages,
                                     std::string heap_set_id,
                                     Component::IBlock_device * block_device,
                                     bool force_init) override
  {
    assert(block_device);
    
    Component::IPager * pager = static_cast<Component::IPager*>
      (new Simple_pager_component(nr_pages, heap_set_id, block_device, force_init));
    
    pager->add_ref();
    block_device->add_ref();
    return pager;
  }
};
  

#endif
