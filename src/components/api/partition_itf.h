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

#ifndef __API_PARTITION_ITF__
#define __API_PARTITION_ITF__

#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus

#include <cstdint>
#include <component/base.h>
#include "memory_itf.h"
#include "block_itf.h"


namespace Component
{

class IPartitioned_device;

/** 
 * Factory for creating partition devices
 * 
 */
class IPartitioned_device_factory : public Component::IBase
{
public:  
  DECLARE_INTERFACE_UUID(0xfac5d826,0x9af6,0x451b,0xbf03,0xd8,0x75,0x11,0xef,0x54,0xc7);

  /** 
   * Instantiate a Paritioned_device component and attach to lower layer
   * 
   * @param block_device Lower layer interface
   * 
   * @return Pointer to IPartitioned_device
   */
  virtual IPartitioned_device* create(IBlock_device * block_device) = 0;
};

/** 
 * IPartitioned_device interface is a single local or remote device
 * and has the global view of all partitions
 * 
 */
class IPartitioned_device : public Component::IBase
{
public:  
  DECLARE_INTERFACE_UUID(0x8155d826,0x9af6,0x451b,0xbf03,0xd8,0x75,0x11,0xef,0x54,0xc7);

  /** 
   * Open a partition and get the block device interface
   * 
   * @param partition_id Partition identifier
   * 
   * @return Block device interface
   */
  virtual Component::IBlock_device * open_partition(unsigned partition_id) = 0;

  /** 
   * Release a partition
   * 
   * @param block_device Block device interface to release 
   */
  virtual void release_partition(IBlock_device * block_device) = 0;


  /** 
   * Get partition information
   * 
   * @param partition_id Partition ID
   * @param size [out] Size of partition in bytes
   * @param part_type [out] Partition type string
   * @param description [out] Description string
   * 
   * @return true if partition exists
   */
  virtual bool get_partition_info(unsigned partition_id, size_t& size, std::string& part_type, std::string& description) = 0;
};


} // namespace Component

#endif 

#endif
