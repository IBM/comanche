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

#ifndef __NVME_DEVICE_H__
#define __NVME_DEVICE_H__

#if defined(__cplusplus)

#include <vector>
#include <common/exceptions.h>
#include <spdk/nvme.h>
#include <core/ring_buffer.h>
#include <core/poller.h>

#include "types.h"
#include "nvme_buffer.h"
#include "nvme_init.h"
#include "nvme_queue.h"

//#define CONFIG_USE_CRUNTIME_DESC_MGT /*< use new/delete instead of ring buffer */

#ifndef NUMA_SOCKET_ANY
#define NUMA_SOCKET_ANY -1
#endif

enum {
  OP_FLAG_READ=0x2,
  OP_FLAG_WRITE=0x4,
};

class Nvme_queue;

class Nvme_device
{
public:
  static constexpr unsigned IO_SW_QUEUE_DEPTH    = 2048; /**< must be power of 2 */
  static constexpr unsigned DEFAULT_NAMESPACE_ID = 1;
  static constexpr bool option_DEBUG = false;
  
  /**
   * General exception type
   *
   */
  class Device_exception : public Exception
  {
  public:
    Device_exception(const char *cause) : Exception(cause)
    {
    }
  };

public:

  /** 
   * Constructor
   * 
   * @param device_id PCI device identifier, e.g., 8b:00.0
   * @param mode Operational mode (MODE_DIRECT, MODE_QUEUED)
   * @param io_threads Core mask for IO threads (MODE_QUEUED only)
   */
  Nvme_device(const char * device_id,
              cpu_mask_t& io_thread_mask);
  
  Nvme_device(const char * device_id,
              Core::Poller * poller);
  

  /**
   * Destructor
   *
   */
  ~Nvme_device();

  /**
   * Allocate IO queues and attach to namespace specified
   *
   * @param namespace_id Namespace identifier
   */
  Nvme_queue * allocate_io_queue_pair(uint32_t namespace_id = 1);
  
  /** 
   * Allocate a buffer for the queue. Round up to block size.
   * 
   * @param num_bytes 
   * @param zero_init 
   * @param numa_socket 
   * 
   * @return 
   */
  void * allocate_io_buffer(size_t num_bytes, bool zero_init, int numa_socket = -1);

  /** 
   * Free previously allocated IO buffer
   * 
   * @param buffer Pointer to memory alloacted with allocate_io_buffer
   */
  void free_io_buffer(void * buffer);
  
  /**
   * Get the block size of the device. Usually 512 or 4096 depending on
   * formatting.
   *
   * @param namespace_id Namespace identifier (default=1)
   *
   * @return Device block size in bytes
   */
  size_t get_block_size(uint32_t namespace_id = 1);

  /**
   * Get the size of the namespace in logical blocks.
   *
   * @param namespace_id Namespace identifier (default=1)
   *
   * @return Size of namespace in blocks
   */
  size_t get_size_in_blocks(uint32_t namespace_id = 1);

  /**
   * Get the size of a namespace in bytes
   *
   * @param namespace_id
   *
   * @return
   */
  size_t get_size_in_bytes(uint32_t namespace_id = 1);

  /** 
   * Get maximum submission queue depth for device
   * 
   * @param namespace_id Namespace number
   * 
   * @return 
   */
  size_t get_max_squeue_depth(uint32_t namespace_id = 1);

  /** 
   * Get device identifier
   * 
   * 
   * @return Volume description
   */
  const char * get_device_id();

  /** 
   * Get PCI identifier
   * 
   * 
   * @return PCI identifier
   */
  const char * get_pci_id();

  /**
   * Low-level format of the device
   *
   * @param block_size Size of blocks in bytes
   * @param namespace_id Namespace identifier
   */
  void format(unsigned lbaf, uint32_t namespace_id = 1);

  /**
   * Perform self tests
   *
   * @param testname Test name
   * @param namespace_id Namespace identifier
   */
  void self_test(std::string testname, uint32_t namespace_id = 1);

  /**
   * Perform a raw block read and dump data to stdout
   *
   * @param lba Block address
   */
  void raw_read(unsigned long lba, uint32_t namespace_id = 1);

  /** 
   * Asynchronous IO for queued mode
   * 
   * @param buffer Buffer to perform IO operation on
   * @param lba Logical block address
   * @param lba_count Logical block count
   * @param op Operation
   * @param tag Tag
   * @param cb Callback for completion
   * @param arg0 Optional argument for callback   
   * @param arg1 Optional second argument for callback
   * 
   */
  void queue_submit_async_op(void *buffer,
                             uint64_t lba,
                             uint64_t lba_count,
                             int op,
                             uint64_t tag,
                             io_callback_t cb,
                             void *arg0 = nullptr,
                             void *arg1 = nullptr,
                             int queue_id = 0);

  /** 
   * Check if all operations up to and inclusive of this gwid 
   * are complete.
   * 
   * @param gwid 
   * 
   * @return 
   */
  bool check_completion(uint64_t gwid, int queue_id = 0);

  /** 
   * Determine if any operations are still pending.
   * 
   * 
   * @return 
   */
  bool pending_remain();
  
  /** 
   * Get ctrlr capabilities
   * 
   * 
   * @return 
   */
  const struct spdk_nvme_ctrlr_data * get_controller_caps();


  /** 
   * Get max IO transfer size
   * 
   * 
   * @return Max transfer size in bytes
   */
  uint32_t get_max_io_xfer_size();

  /** 
   * Get meta data size in bytes
   * 
   * 
   * @return Metadata size in bytes
   */
  uint32_t get_metadata_size();

  /** 
   * Get device namespace flags
   * 
   * 
   * @return flags
   */
  uint32_t get_ns_flags();

  /** 
   * Get hash of serial number of device
   * 
   * 
   * @return Stringified serial number
   */
  uint64_t get_serial_hash();


  /** 
   * Attach a work function to a queue's polling thread
   * 
   * @param queue_id Queue identifier
   * @param work_function Work function pointer
   * @param arg Argument to pass to work function
   */
  void attach_work(unsigned queue_id, std::function<void(void*)> work_function, void * arg);
  
  /* helpers */
  inline struct spdk_nvme_ns * ns() const
  { assert(_probed_device.ns); return _probed_device.ns; }
  
  inline struct spdk_nvme_ctrlr * controller()
  { assert(_probed_device.ctrlr); return _probed_device.ctrlr; }

  inline const struct spdk_nvme_ns_data * ns_data()
  { assert(_probed_device.ns); return spdk_nvme_ns_get_data(_probed_device.ns); }

  inline bool test_exit_io_threads() const
  { return _exit_io_threads; }

  
public:

  void register_ring(unsigned core, struct rte_ring * ring) {
    std::lock_guard<std::mutex> g(_qm_state.lock);
    _qm_state.ring_list[core] = ring;
    _qm_state.launched++;
    wmb();
    PLOG("registering queue message ring (core=%u) : %p", core, ring);
  }

  void free_desc(IO_descriptor * desc) {
#ifdef CONFIG_USE_CRUNTIME_DESC_MGT
    delete desc;
#else
    _desc_ring.mp_enqueue(desc);
#endif
  }
  
  IO_descriptor * alloc_desc() {
#ifdef CONFIG_USE_CRUNTIME_DESC_MGT
    return new IO_descriptor;
#else
    IO_descriptor * desc = nullptr;

    unsigned retries = 0;
    while(!desc) {
      _desc_ring.mc_dequeue(desc);
      if(!desc) usleep(100); /* back off */
      if(retries++ > 1000)
        PWRN("descriptor ring empty");
    }
    assert(desc);
    return desc;
#endif
  }

  size_t queue_count() const {
    return _queues.size();
  }

private:
  /* we managed descriptors at the device level because there is 
     no 1:1 mapping of sw to nvme queues 
  */
  Core::Ring_buffer<IO_descriptor*> _desc_ring;

  static constexpr unsigned MAX_IO_QUEUES = 64;
  static constexpr size_t DESC_RING_SIZE = IO_SW_QUEUE_DEPTH * MAX_IO_QUEUES;


  /**< per worker IO queue */
  struct
  {
    std::mutex       lock;
    struct rte_ring* ring_list[MAX_IO_QUEUES];
    unsigned         launched = 0;
  } _qm_state;

  
  /**
   * Initialize memory, probe devices, attach and initialize hw
   *
   *
   * @return
   */
  void initialize(const char *device_id);
  struct spdk_nvme_ns *  get_namespace(uint32_t namespace_id);

  int                            _mode; /**< operation mode */
  bool                           _exit_io_threads;
  bool                           _activate_io_threads;
  std::vector<Nvme_queue *>      _queues;
  std::vector<unsigned>          _cores;
  struct probed_device           _probed_device;
  unsigned                       _default_core = 0;
  std::string                    _pci_id;
  std::string                    _volume_id;
};


#else
#error Header not supported for plain C
#endif

#endif
