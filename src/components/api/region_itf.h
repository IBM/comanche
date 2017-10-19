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

#ifndef __API_REGION_ITF__
#define __API_REGION_ITF__

#include <common/types.h>
#include <api/components.h>
#include <api/block_itf.h>

namespace Component
{


/** 
 * IRegion_manager is a interface for lightweight partitioning of a block space.
 * Partitions are uniquely identified through owner,id string pairs.
 */
class IRegion_manager : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0x15db1848,0x321a,0x4f85,0x898b,0xff,0xdf,0x12,0x4a,0x23,0x70);

public:
  typedef struct {
    size_t size_in_blocks;
    size_t block_size;
  } REGION_INFO;

  /** 
   * Re-use or allocate a region of space
   * 
   * @param size_in_blocks Size of region in blocks
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
                                                              bool& reused) = 0;
  
  /** 
   * Retrieve region information
   * 
   * @param index Index counting from 0. Note entry may not be occupied.
   * @param ri REGION_INFO [out]
   * 
   * @return true if valid and REGION_INFO has been filled
   */
  virtual bool get_region_info(unsigned index, REGION_INFO& ri) = 0;

  /** 
   * Retrieve region information (alternative API)
   * 
   * @param owner Owner
   * @param id Identifier
   * @param ri REGION_INFO [out]
   * 
   * @return true if valid and REGION_INFO has been filled
   */
  virtual bool get_region_info(std::string owner, std::string id, REGION_INFO& ri) = 0;

  /** 
   * Delete a region (owner,id) pair are unique
   * 
   * @param owner Owner
   * @param id Identifier
   */
  virtual bool delete_region(std::string owner, std::string id) = 0;

  /** 
   * Get block size in bytes
   * 
   * 
   * @return Block size in bytes
   */
  virtual size_t block_size() = 0;

  /** 
   * Return number of regions marked as allocated
   * 
   * 
   * @return Number of allocated regions
   */
  virtual size_t num_regions() = 0;


  /** 
   * Get volume infor for underlying block device
   * 
   * @param vi [out] VOLUME_INFO data structure
   */
  virtual void get_underlying_volume_info(Component::VOLUME_INFO& vi) = 0;

  /**
   * Get underlying block device*
   */
  virtual Component::IBlock_device * get_underlying_block_device() = 0;
  
};


class IRegion_manager_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfacb1848,0x321a,0x4f85,0x898b,0xff,0xdf,0x12,0x4a,0x23,0x70);


  enum {
    FLAGS_CREATE = 0x1,
    FLAGS_FORMAT = 0x2, /*< force region manager to format block device */
  };

  /** 
   * Open a region managed block device
   * 
   * @param block_device Underlying block device
   * @param flags Instantiation flags
   * 
   * @return Pointer to IRegion_manager interface
   */
  virtual IRegion_manager * open(Component::IBlock_device * block_device, int flags) = 0;

};


}


#endif 
