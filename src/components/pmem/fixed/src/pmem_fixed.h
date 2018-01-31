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

#ifndef __PMEM_FIXED_COMPONENT_H__
#define __PMEM_FIXED_COMPONENT_H__

#include <common/spinlocks.h>
#include <api/block_itf.h>
#include <api/pmem_itf.h>
#include <api/pager_itf.h>
#include <api/region_itf.h>

#include <string>
#include <list>

class Pmem_fixed_component : public Component::IPersistent_memory
{  
private:
  static constexpr bool option_DEBUG = true;


public:
  Pmem_fixed_component(std::string owner_id,
                       Component::IBlock_device * block_device,
                       bool force_init);

  Pmem_fixed_component(std::string owner_id,
                       Component::IRegion_manager * rm,
                       bool force_init);  
  
  /** 
   * Destructor
   * 
   */
  virtual ~Pmem_fixed_component();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x21562849,0xb0ea,0x41a3,0xaed8,0x8f,0x9d,0xfb,0x4a,0xf4,0xaa);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IPersistent_memory::iid()) {
      return (void *) static_cast<Component::IPersistent_memory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

public:

  /* IPersistent_memory */

  
  /** 
   * Allocate a region of persistent memory (block device backed)
   * 
   * @param id Unique region identifier
   * @param size Size of virtual space in bytes
   * @param numa_node NUMA node specifier
   * @param vptr [out] virtual address pointer
   * 
   * @return Handle
   */
  virtual pmem_t open(std::string id,
                      size_t size,
                      int numa_node,
                      bool& reused,
                      void*& vptr) override;
  

  /** 
   * Free a previously allocated persistent memory region
   * 
   * @param handle Handle to persistent memory region
   * @param flags Flags
   */
  virtual void close(pmem_t handle, int flags = 0) override;


  /** 
   * Erase a region of memory
   * 
   * @param handle 
   */
  virtual void erase(pmem_t handle) override;


  /** 
   * Determine if a virtual address is in persistent memory
   * 
   * @param p Virtual address pointer
   * 
   * @return True if belonging to persistent memory
   */
  virtual bool is_pmem(void * p) override;

  /** 
   * Flush all volatile data to peristent memory in a non-transacation context.
   * 
   * @param handle Persistent memory handle
   */
  virtual void persist(pmem_t handle) override;

  /** 
   * Flush a specific region of memory
   * 
   * @param handle Persistent memory handle
   * @param ptr Virtual address
   * @param size Size in bytes
   */
  virtual void persist_scoped(pmem_t handle, void *ptr, size_t size) override;

  /** 
   * Get size of persistent memory in bytes
   * 
   * 
   * @return Size in bytes
   */
  virtual size_t get_size(pmem_t handle) override;

  /** 
   * Get the write-atomicity size for the underlying device.
   * 
   * 
   * @return Atomicity size in bytes (usually 4096)
   */
  virtual size_t get_atomicity_size() override {
    return _block_size;
  }
  
  /** 
   * Start transaction
   * 
   */
  virtual void tx_begin(pmem_t handle) override;

  /** 
   * Commit transaction
   * 
   * 
   * @return S_OK or E_FAIL
   */
  virtual status_t tx_commit(pmem_t handle) override;
  

private:

  std::string                         _owner_id;
  int                                 _fd_xms;
  Component::IBlock_device *          _block_device = nullptr;
  Component::IRegion_manager *        _rm;
  size_t                              _block_size;
  

  /* we use a structure in the handle so that we
     can avoid cross-thread locking */
  struct mem_handle {
    Component::io_buffer_t     iob;
    size_t                     size;
    void *                     ptr;
    Component::IBlock_device * bd;
  };

  std::list<struct mem_handle*>       _handle_list;
  std::mutex                          _handle_list_lock;
};


class Pmem_fixed_component_factory : public Component::IPersistent_memory_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac62849,0xb0ea,0x41a3,0xaed8,0x8f,0x9d,0xfb,0x4a,0xf4,0xaa);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IPersistent_memory_factory::iid()) {
      return (void *) static_cast<Component::IPersistent_memory_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  virtual Component::IPersistent_memory * open_allocator(std::string owner_id,
                                                         Component::IBlock_device * block_device,
                                                         bool force_init) override
  {
    using namespace Component;
    IPersistent_memory * pm = static_cast<IPersistent_memory*>
      (new Pmem_fixed_component(owner_id, block_device, force_init));
    block_device->add_ref();
    pm->add_ref();
    return pm;
  }

  virtual Component::IPersistent_memory * open_allocator(std::string owner_id,
                                                         Component::IRegion_manager * rm,
                                                         bool force_init) override
  {
    using namespace Component;
    IPersistent_memory * pm = static_cast<IPersistent_memory*>
      (new Pmem_fixed_component(owner_id, rm, force_init));

    pm->add_ref();
    return pm;
  }

};



#endif
