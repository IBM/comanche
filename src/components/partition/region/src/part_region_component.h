#pragma once

#ifndef __PART_REGION_COMPONENT_H__
#define __PART_REGION_COMPONENT_H__

#include <api/block_itf.h>
#include <api/region_itf.h>
#include <string>
#include <list>
#include "region_table.h"

class Region_session;

class Region_manager : public Component::IRegion_manager
{
private:
  static constexpr bool option_DEBUG = true;
  
public:
  Region_manager(Component::IBlock_device * block_device, bool force_init);
  
  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x7d55365a,0xeccc,0x4e13,0x9281,0xaa,0xc9,0xfc,0x5b,0xc5,0xa9);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IRegion_manager::iid()) {
      return (void *) static_cast<Component::IRegion_manager*>(this);
    }
    else
      return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }
  
  /** 
   * Destructor
   * 
   */
  virtual ~Region_manager();

public:
  /** 
   * IRegion_manager
   */
  
  /** 
   * Re-use or allocate a region of space
   * 
   * @param size Size of region in blocks
   * @param owner Owner
   * @param id Identifier
   * @param vaddr [out] Virtual load address
   * @param reused [out] true if re-used.
   * 
   * @return Block device interface onto region
   */
  virtual Component::IBlock_device * reuse_or_allocate_region(size_t size_in_blocks,
                                                              std::string owner,
                                                              std::string id,
                                                              addr_t& vaddr,
                                                              bool& reused) override;
  
  /** 
   * Retrieve region information
   * 
   * @param index Index counting from 0
   * @param ri REGION_INFO [out]
   * 
   * @return true if valid and REGION_INFO has been filled
   */
  virtual bool get_region_info(unsigned index, REGION_INFO& ri) override;

  /** 
   * Retrieve region information (alternative API)
   * 
   * @param owner Owner
   * @param id Identifier
   * @param ri REGION_INFO [out]
   * 
   * @return true if valid and REGION_INFO has been filled
   */
  virtual bool get_region_info(std::string owner, std::string id, REGION_INFO& ri) override;

  /** 
   * Delete a region
   * 
   * @param index Region index
   */
  virtual bool delete_region(std::string owner, std::string id) override;
  
  /** 
   * Get block size in bytes
   * 
   * 
   * @return Block size in bytes
   */
  virtual size_t block_size() override;

   /** 
   * Return number of allocated regions
   * 
   * 
   * @return Number of allocated regions
   */
  virtual size_t num_regions() override;

  /** 
   * Get volume info for underlying block store
   * 
   * @param vi 
   */
  virtual void get_underlying_volume_info(Component::VOLUME_INFO& vi) override;

  /**
   * Get underlying block device*
   */
  virtual Component::IBlock_device * get_underlying_block_device() override {
    assert(_lower_block_layer);
    return _lower_block_layer;
  }
  
public:
  
  Component::IBlock_device * _lower_block_layer; /*< lower interface connection */
  Component::VOLUME_INFO     _vi;
  Region_table               _region_table; /*< main data structure */
  std::list<Region_session*> _sessions; /*< active sessions/block device windows */
};


class Region_manager_factory : public Component::IRegion_manager_factory
{
public:
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac2ccc2,0x28bb,0x41be,0x8b43,0x07,0xaa,0x90,0x76,0xf8,0x84);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IRegion_manager_factory::iid()) {
      return (void *) static_cast<Component::IRegion_manager_factory*>(this);
    }
    else
      return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  /** 
   * Open a region managed block device
   * 
   * @param block_device Underlying block device
   * @param flags Instantiation flags
   * 
   * @return Pointer to IRegion_manager interface
   */
  virtual Component::IRegion_manager * open(Component::IBlock_device * block_device, int flags) override;

};


#endif
