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
#pragma once

#ifndef __PART_GPT_COMPONENT_H__
#define __PART_GPT_COMPONENT_H__

#include <api/block_itf.h>
#include <api/partition_itf.h>
#include <string>
#include <list>

#include "gpt.h"

class GPT_component_factory : public Component::IPartitioned_device_factory
{
public:
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac2ccc2,0x28bb,0x41be,0x8b43,0x07,0xaa,0x90,0x76,0xf8,0x84);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IPartitioned_device_factory::iid()) {
      return (void *) static_cast<Component::IPartitioned_device_factory*>(this);
    }
    else
      return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  /** 
   * Instantiate a Paritioned_device component and attach to lower layer
   * 
   * @param block_device Lower layer interface
   * 
   * @return Pointer to IPartitioned_device
   */
  virtual Component::IPartitioned_device* create(Component::IBlock_device * block_device) override;
};


class Partition_session;

class GPT_component : public Component::IPartitioned_device
{
private:
  static constexpr bool option_DEBUG = true;
  
public:
  GPT_component(Component::IBlock_device * block_device);
  
  /** 
   * Component/interface management
   * 
   */
  DECLARE_COMPONENT_UUID(0xda92ccc2,0x28bb,0x41be,0x8b43,0x07,0xaa,0x90,0x76,0xf8,0x84);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IPartitioned_device::iid()) {
      return (void *) static_cast<Component::IPartitioned_device*>(this);
    }
    else
      return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }
  
public:
  /** 
   * Destructor
   * 
   */
  virtual ~GPT_component();


  /** 
   * IPartitioned_device interface
   */

  
  /** 
   * Open a partition and get the block device interface
   * 
   * @param partition_id Partition identifier
   * 
   * @return Block device interface
   */
  virtual Component::IBlock_device * open_partition(unsigned partition_id) override;

  /** 
   * Release a partition
   * 
   * @param block_device Block device interface to release 
   */
  virtual void release_partition(Component::IBlock_device * block_device) override;

  /** 
   * Get partition information
   * 
   * @param partition_id Partition ID
   * @param size [out] Size of partition in bytes
   * @param part_type [out] Partition type string
   * @param description [out] Description string
   * 
   * @return 
   */
  virtual bool get_partition_info(unsigned partition_id,
                                  size_t& size,
                                  std::string& part_type,
                                  std::string& description) override;

public:
  
  GPT::Partition_table          _table;
  Component::IBlock_device *    _lower_block_layer;
  std::list<Partition_session*> _sessions;
};


#endif
