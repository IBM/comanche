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
#ifndef __BLOCK_POSIX_COMPONENT_H__
#define __BLOCK_POSIX_COMPONENT_H__

#include <aio.h>
#include <string>
#include <mutex>
#include <list>

#include <core/physical_memory.h>
#include <api/partition_itf.h>
  

/** 
 * POSIX-based block device component. Uses O_DIRECT.  We use DPDK
 * to allocate contiguous, pinned memory
 * 
 */
class Block_posix : public Core::Physical_memory,
                    public Component::IBlock_device
{
private:
  static constexpr bool   option_DEBUG = false;
  static constexpr size_t AIO_THREADS = 8;
  static constexpr size_t AIO_SIMUL = 256;
  static constexpr size_t AIO_DESCRIPTOR_POOL_SIZE = 1024;
  static constexpr size_t IO_BLOCK_SIZE = 4096;
  
public:

  /** 
   * Constructor
   * 
   * @param config Configuration string (JSON) of the form 
   * e.g., {"path":"/tmp/foo", "size_in_blocks", 256 }
   * @param size_in_blocks Size in 4KB blocks 
   * 
   */
  Block_posix(std::string config);
  
  /** 
   * Destructor
   * 
   */
  virtual ~Block_posix();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x1a2f047a,0x329c,0x4452,0xa75e,0x5a,0x1c,0xa2,0xe8,0xe4,0xc4);

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
  
  INLINE_FORWARDING_MEMORY_METHODS;
  
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
   * @param lba block address
   * @param lba_count number of blocks
   * @param queue_id Logical queue identifier (counting from 0, -1=unspecified)
   * @param cb Optional call back for completion of IO
   * @param cb_arg0 Optional call back argument
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
   * Synchronous write (busy waits)
   * 
   * @param buffer IO buffer
   * @param offset Offset in logical blocks from the start of the IO buffer
   * @param lba logical block address
   * @param lba_count number of blocks
   * @param queue_id Logical queue identifier (counting from 0, -1=unspecified)
   */
  virtual void write(Component::io_buffer_t buffer,
                     uint64_t buffer_offset,
                     uint64_t lba,
                     uint64_t lba_count,
		     int queue_id = 0) override;
  

  /** 
   * Synchronous read (busy waits)
   * 
   * @param buffer IO buffer
   * @param offset Offset in logical blocks from the start of the IO buffer
   * @param lba logical block address
   * @param lba_count number of blocks
   * @param queue_id Logical queue identifier (counting from 0, -1=unspecified)
   */
  virtual void read(Component::io_buffer_t buffer,
                    uint64_t buffer_offset,
                    uint64_t lba,
                    uint64_t lba_count,
		    int queue_id = 0) override;
  
  /** 
   * Check for completion of a work request. This API is thread-safe.
   * 
   * @param gwid Work request identifier
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
  
  struct aiocb * allocate_descriptor();  
  void free_descriptor(struct aiocb * desc);
  uint64_t add_outstanding(struct aiocb * desc);
  bool check_complete(uint64_t workid);
  
private:
  typedef struct {
    struct aiocb* aiocb;
    uint64_t tag;
    uint32_t magic;
  } work_desc_t;

  std::string                _file_path;
  size_t                     _size_in_blocks;
  int                        _fd;
  std::mutex                 _aiob_vector_lock;
  std::vector<struct aiocb*> _aiob_vector;
  int                        _fd_xms = 0;
  std::mutex                 _work_lock;
  uint64_t                   _work_id;
  std::list<work_desc_t>     _outstanding;
};


/** 
 * Factory class for POSIX-based block device component
 * 
 */
class Block_posix_factory : public Component::IBlock_device_factory
{
public:
  DECLARE_COMPONENT_UUID(0xfacf047a,0x329c,0x4452,0xa75e,0x5a,0x1c,0xa2,0xe8,0xe4,0xc4);
                         
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

  /** 
   * Basic instance creation
   * 
   * @param config Configuration string
   * @param cpuset Optional CPU set for IO threads
   * @param poller Optional poller for shared threading
   * 
   * @return 
   */
  virtual Component::IBlock_device * create(std::string config,
                                            cpu_mask_t * cpuset = nullptr,
                                            Core::Poller * poller = nullptr) override;
  
};

#endif
