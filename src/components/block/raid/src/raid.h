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


/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __RAID_COMPONENT_H__
#define __RAID_COMPONENT_H__

#include <core/physical_memory.h>
#include <api/raid_itf.h>
#include <api/block_itf.h>

/** 
 * RAID component
 * 
 */
class Raid_component : public Component::IRaid,
                       public Core::Physical_memory
{
private:
  static constexpr bool option_DEBUG = false;

  INLINE_FORWARDING_MEMORY_METHODS;
  
public:

  Raid_component();
  virtual ~Raid_component();
  
  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x31197cbd,0xe4c8,0x471b,0x9296,0xb9,0x45,0xbf,0x6b,0xca,0xdb);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IRaid::iid()) {
      return (void *) static_cast<Component::IRaid*>(this);
    }
    else
      return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }
  
public:

  /** 
   * Configure RAID
   * 
   * @param json_configuration Configuration string
   */
  virtual void configure(std::string json_configuration) override;


  /** 
   * Add block device to RAID array
   * 
   * @param device Block device
   * @param role JSON-defined role
   */
  virtual void add_device(Component::IBlock_device * device, std::string role) override;

  /** 
   * Submit asynchronous read operation
   * 
   * @param buffer IO buffer
   * @param offset Offset in logical blocks from the start of the IO buffer
   * @param lba block address
   * @param lba_count number of blocks
   * @param queue_id Logical queue identifier (counting from 0, -1=unspecified)
   * @param cb Optional call back for completion of IO
   * @param cb_arg0 Optional call back argument (normally this)
   * @param cb_arg1 Optional second call back argument
   * 
   * @return Work identifier which is an ordered sequence number
   */
  virtual Component::workid_t async_read(Component::io_buffer_t buffer,
                                         uint64_t buffer_offset,
                                         uint64_t lba,
                                         uint64_t lba_count,
                                         int queue_id = 0,
                                         io_callback_t cb = nullptr,
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
   * @param cb_arg0 Optional call back argument (normally this)
   * @param cb_arg0 Optional second call back argument
   * 
   * @return Work identifier which is an ordered sequence number
   */
  virtual Component::workid_t async_write(Component::io_buffer_t buffer,
                                          uint64_t buffer_offset,
                                          uint64_t lba,
                                          uint64_t lba_count,
                                          int queue_id = 0,
                                          io_callback_t cb = nullptr,
                                          void * cb_arg0 = nullptr,
                                          void * cb_arg1 = nullptr) override;

  /** 
   * Check for completion of a work request AND all prior work
   * requests. This API is thread-safe.  The implementation of this
   * method should not require the async-issuing thread to call
   * check_completion for all issues ops.
   * 
   * @param gwid Work request identifier
   * @param queue_id Logical queue identifier (counting from 0, -1=unspecified)
   * 
   * @return True if completed.
   */
  virtual bool check_completion(Component::workid_t gwid, int queue_id = 0) override;

  /** 
   * Get device information
   * 
   * @param devinfo pointer to VOLUME_INFO struct
   * 
   * @return S_OK on success
   */
  virtual void get_volume_info(Component::VOLUME_INFO& devinfo) override;

  /** 
   * Convert logical gwid (embedding device id) to sequential gwid
   * 
   * @param lgwid Logical gwid
   * 
   * @return Sequential gwid
   */
  virtual uint64_t gwid_to_seq(uint64_t gwid) override;

private:

  static constexpr size_t MAX_DEVICE_COUNT = 16;
  /** 
   * Mapping from lba to device.  Simple modulo.
   * 
   * @param lba Logic block address
   * 
   * @return 
   */
  inline IBlock_device * select_device(uint64_t lba, unsigned& index)
  {
    assert(lba <= _logical_block_count);
    index = lba % _device_count;
    return _bdv_itf[index].block_device;
  }

  struct __device {
    Component::IBlock_device* block_device;
    unsigned                  flags; /* placeholder for role etc. */
  };

  int                   _raid_level = -1;
  std::vector<__device> _bdv_itf;
  bool                  _ready = false;
  unsigned              _device_count = 0;
  size_t                _block_count = 0;
  size_t                _logical_block_count = 0;
};


#endif
