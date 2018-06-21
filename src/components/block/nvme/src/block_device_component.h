/*
   Copyright [2017] [IBM Corporation]

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
#ifndef __COMANCHE_BLOCK_DEVICE_H__
#define __COMANCHE_BLOCK_DEVICE_H__

#include "nvme_device.h"
#include <core/physical_memory.h>
#include <core/poller.h>
#include <api/block_itf.h>
//#include <api/memory_itf.h>
#include "types.h"

using namespace Component;

/** 
 * Block_device: provides unified API for NVMe device access.
 * Incorporates Block_service and Block_service_session.  Current
 * version is simple, and supports a single session. 
 * 
 */
class Block_device_component: public Core::Physical_memory,
                              public Component::IBlock_device
{
public:

  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x24518582,0xd731,0x4406,0x9eb1,0x58,0x70,0x26,0x40,0x8e,0x23);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == IBlock_device::iid()) {
      return (void *) static_cast<IBlock_device *>(this);
    }
    else 
      return NULL; // we don't support this interface
  };

  void unload() override {
    delete this;
  }

  INLINE_FORWARDING_MEMORY_METHODS;

  
public:
  
  /** 
   * Constructor
   * 
   * @param pci_addr 
   * @param cpus 
   * @param poller 
   * 
   * @return 
   */
  Block_device_component(const char * pci_addr, cpu_mask_t* cpus, Core::Poller * poller);

  /**
   * Default destructor
   */
  virtual ~Block_device_component() noexcept;

  /** 
   * Submit asynchronous read operation
   * 
   * @param buffer IO buffer
   * @param offset Offset in logical blocks from the start of the IO buffer
   * @param lba block address
   * @param lba_count number of blocks
   * @param queue_id Logical queue identifier (counting from 0, -1=unspecified)
   * @param cb Optional call back for completion of IO
   * @param cb_arg0 Optional call back argument
   * @param cb_arg1 Optional second call back argument
   * 
   * @return Work identifier which is an ordered sequence number
   */
  virtual workid_t async_read(io_buffer_t buffer,
                              uint64_t offset,
                              uint64_t lba,
                              uint64_t lba_count,
                              int queue_id = 0,
                              io_callback_t callback = nullptr,
                              void * cb_arg0 = nullptr,
                              void * cb_arg1 = nullptr) override;

  /** 
   * Submit asynchronous write operation
   * 
   * @param buffer IO buffer
   * @param offset Offset in logical blocks from the start of the IO buffer
   * @param lba logical block address
   * @param lba_count number of blocks
   * @param queue_id Optional queue identifier counting from 0
   * @param cb Optional call back for completion of IO
   * @param cb_arg0 Optional call back argument
   * @param cb_arg1 Optional second call back argument
   * 
   * @return Work identifier which is an ordered sequence number
   */
  virtual workid_t async_write(io_buffer_t buffer,
                               uint64_t offset,
                               uint64_t lba,
                               uint64_t lba_count,
                               int queue_id = 0,
                               io_callback_t callback = nullptr,
                               void * cb_arg0 = nullptr,
                               void * cb_arg1 = nullptr) override;

  /** 
   * Attach a work function to be called by IO servicing threads
   * 
   * @param queue_id Queue identifier
   * @param work_function Work function pointer
   * @param arg Argument to pass to work function
   */
  virtual void attach_work(std::function<void(void*)> work_function,
                           void * arg,
                           int queue_id = 0) override;

  /** 
   * Check for completion of a work request. This API is thread-safe.
   * 
   * @param gwid Work request identifier
   * 
   * @return True if completed.
   */
  virtual bool check_completion(uint64_t gwid, int queue_id = 0) override;


  /** 
   * Get device information
   * 
   * @param devinfo pointer to VOLUME_INFO struct
   * 
   * @return S_OK on success
   */
  virtual void get_volume_info(Component::VOLUME_INFO& devinfo) override;

private:

  static constexpr unsigned DEFAULT_NAMESPACE_ID = 1;
  static constexpr unsigned NUM_HW_IO_QUEUES = 1;
  
  Nvme_device * _device;
  
}; 

/** 
 * Factory for Block_device component
 * 
 */
class Block_device_component_factory : public Component::IBlock_device_factory
{
public:
  DECLARE_COMPONENT_UUID(0xFAC2215b,0xff57,0x4efd,0x9e4e,0xca,0xe8,0x9b,0x70,0x73,0x2a);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == IBlock_device_factory::iid()) {
      return (void *) static_cast<IBlock_device_factory *>(this);
    }
    else 
      return NULL; // we don't support this interface
  };

  void unload() override {
    delete this;
  }
  
  IBlock_device * create(std::string device_id,
                         cpu_mask_t * cpuset = nullptr,
                         Core::Poller * poller = nullptr) override {

    IBlock_device * inst = static_cast<Component::IBlock_device *>
      (new Block_device_component(device_id.c_str(), cpuset, poller));
    
    inst->add_ref();
    return inst;
  }

};



#endif
