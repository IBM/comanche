#pragma once

#ifndef __PMEM_PAGED_COMPONENT_H__
#define __PMEM_PAGED_COMPONENT_H__

#include <common/spinlocks.h>
#include <api/block_itf.h>
#include <api/pmem_itf.h>
#include <api/pager_itf.h>

#include <string>
#include <list>

class Pmem_paged_component : public Component::IPersistent_memory
{  
private:
  static constexpr bool option_DEBUG = true;


public:
  Pmem_paged_component(std::string owner_id,
                       Component::IPager * pager,
                       bool force_init);
  
  /** 
   * Destructor
   * 
   */
  virtual ~Pmem_paged_component();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x8a239351,0xa08b,0x4d43,0xb4cc,0xf4,0x88,0x92,0xac,0x4f,0x77);
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
   * Open a region of persistent memory (block device backed)
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
   * Explicitly erase a region of memory
   * 
   * @param handle 
   */
  virtual void erase(pmem_t handle) override {
    throw API_exception("%s: not implemented", __PRETTY_FUNCTION__);
  }


  /** 
   * Get number of faults
   * 
   * @return Number of faults
   */
  virtual size_t fault_count() override {
    return _fault_count;
  }
  
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
    return _pager->get_atomicity_size();
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
  

  bool pf_handler(addr_t addr);
  
private:

  std::string                _owner_id;
  int                        _fd_xms;
  Component::IPager *        _pager;
  Component::VOLUME_INFO     _vi;
  uint64_t                   _fault_count __attribute__((aligned(8))) = 0;
};


class Pmem_paged_component_factory : public Component::IPersistent_memory_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac39351,0xa08b,0x4d43,0xb4cc,0xf4,0x88,0x92,0xac,0x4f,0x77);

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
                                                         Component::IPager * pager,
                                                         bool force_init) override
  {
    using namespace Component;
    IPersistent_memory * pm = static_cast<IPersistent_memory*>(new Pmem_paged_component(owner_id, pager, force_init));
    pm->add_ref();
    pager->add_ref();
    return pm;
  }

};



#endif
