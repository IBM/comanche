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

#ifndef __NVME_DEVICE_H__
#define __NVME_DEVICE_H__

#if defined(__cplusplus)

#include <vector>

#include <common/exceptions.h>
#include <spdk/nvme.h>

#include "types.h"
#include "nvme_buffer.h"
#include "nvme_init.h"
#include "nvme_queue.h"


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
  static constexpr unsigned QUEUED_MODE_IOQ_DEPTH = 4096; /**< must be power of 2 */

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
  Nvme_device(const char *device_id,
              int mode = MODE_DIRECT,
              cpu_set_t* io_threads = nullptr);

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
   * @param num_bytes Size of buffer to allocate
 Pointer to new buffer object which must be destroyed by caller.
   */
  void * allocate_io_buffer(size_t num_bytes, bool zero_init, int numa_socket = -1) __attribute__((deprecated));


  /** 
   * Free previously allocated IO buffer
   * 
   * @param buffer Pointer to memory alloacted with allocate_io_buffer
   */
  void free_io_buffer(void * buffer) __attribute__((deprecated));
  
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
   * Synchronous IO for queued mode
   * 
   * @param buffer Buffer to perform IO operation on
   * @param lba 
   * @param lba_count 
   * @param op 
   * 
   * @return 
   */
  status_t queue_submit_sync_op(void *buffer, uint64_t lba, uint64_t lba_count, int op);

  /** 
   * Asynchronous IO for queued mode
   * 
   * @param buffer Buffer to perform IO operation on
   * @param lba Logical block address
   * @param lba_count Logical block count
   * @param op Operation
   * @param tag Tag
   * @param cb Callback for completion
   * @param arg Optional argument for callback
   * 
   */
  void queue_submit_async_op(void *buffer,
                             uint64_t lba,
                             uint64_t lba_count,
                             int op,
                             int tag,
                             io_callback_t cb,
                             void *arg = nullptr);

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
  
  /* helpers */
  inline struct spdk_nvme_ns * ns() const
  { assert(_probed_device.ns); return _probed_device.ns; }
  
  inline struct spdk_nvme_ctrlr * controller()
  { assert(_probed_device.ctrlr); return _probed_device.ctrlr; }

  inline const struct spdk_nvme_ns_data * ns_data()
  { assert(_probed_device.ns); return spdk_nvme_ns_get_data(_probed_device.ns); }

  inline bool exit_io_threads() const
  { return _exit_io_threads; }

  
private:

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
  std::vector<Nvme_queue *>      _queues;
  struct probed_device           _probed_device;
};


#else
#error Header not supported for plain C
#endif

#endif
