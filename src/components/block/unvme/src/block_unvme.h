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

#pragma once

#ifndef __BLOCK_POSIX_COMPONENT_H__
#define __BLOCK_POSIX_COMPONENT_H__

extern "C" {
#include <unvme.h>
}

#include <map>
#include <string>
#include <api/block_itf.h>

/** 
 * Factory class for POSIX-based block device component
 * 
 */
class Block_unvme_factory : public Component::IBlock_device_factory
{
public:
  DECLARE_COMPONENT_UUID(0xface284f,0x9c09,0x4e7d,0x8308,0x3a,0xa4,0xb0,0xc6,0x7d,0xfd);
                         
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IBlock_device_factory::iid()) {
      return (void *) static_cast<Component::IBlock_device_factory*>(this);
    }
    else
      return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  DUMMY_IBASE_CONTROL;

  /** 
   * Instantiate a Paritioned_device component and attach to lower layer
   * 
   * @param block_device Lower layer interface
   * 
   * @return Pointer to IPartitioned_device
   */
  virtual Component::IBlock_device* create(std::string config,
                                           cpu_mask_t * cpus,
                                           Core::Poller * poller) override;
};
  

/** 
 * POSIX-based block device component. Uses O_DIRECT
 * 
 */
class Block_unvme : public Component::IBlock_device
{
private:
  static constexpr bool   option_DEBUG = false;
  static constexpr size_t QUEUE_COUNT = 4;
  static constexpr size_t QUEUE_SIZE = 24;
public:

  /** 
   * Constructor
   * 
   * @param config Configuration string (JSON) of the form 
   * e.g., {"path":"/tmp/foo", "size_in_blocks", 256 }
   * @param size_in_blocks Size in 4KB blocks 
   * 
   */
  Block_unvme(std::string config);
  
  /** 
   * Destructor
   * 
   */
  virtual ~Block_unvme();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x1efe284f,0x9c09,0x4e7d,0x8308,0x3a,0xa4,0xb0,0xc6,0x7d,0xfd);

  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IBlock_device::iid()) {
      return (void *) static_cast<Component::IBlock_device*>(this);
    }
    else
      return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }
  
  DUMMY_IBASE_CONTROL;


public:

  /** 
   * Allocate a contiguous memory region that can be used for IO
   * 
   * @param size Size of memory in bytes
   * @param alignment Alignment in bytes
   * @param numa_node NUMA node (-1) for any
   * 
   * @return Handle to IO memory region
   */
  virtual Component::io_buffer_t allocate_io_buffer(size_t size,
                                                    unsigned alignment,
                                                    int /* numa_node */) override
  {
    if(alignment > 0)
      PWRN("Block_unvme: does not support alignment");
    
    void * ptr = unvme_alloc(_ns, size);
    _len_map[ptr] = size;
    return reinterpret_cast<Component::io_buffer_t>(ptr);
  }

  
  /** 
   * Re-allocate area of memory
   * 
   * @param io_mem Memory handle (from allocate_io_buffer)
   * @param size New size of memory in bytes
   * @param alignment Alignment in bytes
   * 
   * @return S_OK or E_NO_MEM
   */
  virtual status_t realloc_io_buffer(Component::io_buffer_t io_mem,
                                     size_t size,
                                     unsigned alignment) override
  {
    throw API_exception("not implemented");
    return E_NOT_IMPL;
  }

  /** 
   * Free a previously allocated buffer
   * 
   * @param io_mem Handle to IO memory allocated by allocate_io_buffer
   * 
   * @return S_OK on success
   */
  virtual status_t free_io_buffer(Component::io_buffer_t io_mem) override
  {
    void * ptr = reinterpret_cast<void*>(io_mem);
    if(ptr==nullptr)
      throw API_exception("free_io_buffer: bad parameter");
    _len_map.erase(ptr);
    unvme_free(_ns, ptr);
  }

  /** 
   * Register memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual void register_memory_for_io(void * vaddr, size_t len) override
  {
    throw API_exception("not implemented");
  }

  /** 
   * Unregister memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual void unregister_memory_for_io(void * vaddr, size_t len) override
  {
    throw API_exception("not implemented");
  }

  /** 
   * Get pointer (virtual address) to start of IO buffer
   * 
   * @param buffer IO buffer handle
   * 
   * @return pointer
   */
  virtual void * virt_addr(Component::io_buffer_t buffer) override
  {
    return reinterpret_cast<void*>(buffer);
  }

  /** 
   * Get physical address of buffer
   * 
   * @param buffer IO buffer handle
   * 
   * @return physical address
   */
  virtual addr_t phys_addr(Component::io_buffer_t buffer)
  {
    throw API_exception("physical addressing not supported");
  }
  
  /** 
   * Get size of memory buffer
   * 
   * @param buffer IO memory buffer handle
   * 
   * @return 
   */
  virtual size_t get_size(Component::io_buffer_t buffer)
  {
    return _len_map[reinterpret_cast<void*>(buffer)];
  }
  
  /** 
   * Submit asynchronous read operation
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
   * @param cb_arg0 Optional call back argument
   * @param cb_arg1 Optional second call back argument
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
   * Check for completion of a work request. This API is thread-safe.
   * 
   * @param gwid Work request identifier
   * @param queue_id Logical queue identifier (not used)
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

private:
  std::map<void*, size_t> _len_map;
  const unvme_ns_t*       _ns;
  size_t                  _size_in_blocks;

};


#endif
