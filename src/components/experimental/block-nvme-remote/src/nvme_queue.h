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

#ifndef __NVME_QUEUE_H__
#define __NVME_QUEUE_H__

#if defined(__cplusplus)
#include <common/types.h>
#include <pthread.h> 
#include <common/cycles.h>
#include "nvme_device.h"
#include "nvme_queue.h"
#include "config.h"
#include "types.h"

struct spdk_nvme_qpair;

class Nvme_device;
class Nvme_queue;

typedef struct
{
  uint64_t tag;  
  DPDK::Ring_buffer<uint64_t>* list;
  bool used;
  Nvme_queue * nvme_queue;
#ifdef CONFIG_QUEUE_STATS
  cpu_time_t time_stamp;
#endif
} cb_arg;


/**
 * Represents an IO queue pair on a device
 *
 */
class Nvme_queue
{
  friend class NVMe_storage_device;

 private:
  static const bool option_DEBUG = false;
  const unsigned NUM_SUB_FRAMES  = 4096;
  
 public:
  /**
   * Constructor
   *
   * @param device Pointer to Nvme_device instance
   * @param qid Queue identifier
   * @param qpair Pointer to queue pair
   */
  Nvme_queue(Nvme_device *device,
             unsigned qid,
             struct spdk_nvme_qpair *const qpair);

  /**
   * Destructor
   *
   */
  ~Nvme_queue();

  /**
   * Perform a synchronous write (non Nvme_buffer version)
   *
   * @param buffer Buffer to write
   * @param lba Logical block address
   * @param lba_count Number of blocks to write
   * @param op Operation (OP_FLAG_READ, OP_FLAG_WRITE)
   *
   * @return S_OK on success
   */
  status_t submit_sync_op(void *buffer, uint64_t lba, uint64_t lba_count, int op);

  /**
   * Perform a asynchronous read (non Nvme_buffer version).
   *
   * @param buffer Target buffer
   * @param lba Logical block address
   * @param lba_count Number of blocks to write
   * @param op Operation (OP_FLAG_READ, OP_FLAG_WRITE)
   * @param tag Used to decern a group of operations from the caller perspective
   *
   * @return 0 on success
   */
  status_t submit_async_op(void *buffer,
                           uint64_t lba,
                           uint64_t lba_count,
                           int op,
                           uint64_t tag);


  /** 
   * Submit asynchronous IO operation (internal use only)
   * 
   * @param desc IO descriptor
   */
  void submit_async_op_internal(queued_io_descriptor_t * desc);
  
  /** 
   * Get the next completion tag.
   * 
   * 
   * @return Async operation tag or 0 if none available.
   */
  uint64_t get_next_async_completion();
  
  /**
   * Get the number of outstanding operations 
   *
   * @return Number of outstanding IOPs
   */
  size_t outstanding_count();

  /** 
   * Get tag of last known completion
   * 
   * @return ID of last completed operation
   */
  uint64_t get_last_completion();

  /**
   * Process any pending completions
   *
   * @param limit Max number of completions to process; 0 = infinite
   */
  int32_t process_completions(int limit = 0);

  /**
   * Helpers
   *
   */
  inline const Nvme_device *device() { return _device;  }
  inline uint64_t max_lba() const { return _max_lba;  }
  inline unsigned block_size() const { return _block_size; }
  
 private:
  static const size_t ASYNC_STATUS_ARRAY_SIZE = 32;

  Nvme_device *_device;
  struct spdk_nvme_qpair *_qpair;
  unsigned _ns_id;
  uint64_t _max_lba;  // for bounds checking this should be fast to access
  unsigned _block_size;
  unsigned _queue_id;

  /* this does not need to be thread safe - this interface is
     not reentrant since IO queues are not thread safe at the SPDK
     level */
  DPDK::Ring_buffer<uint64_t> _completion_list;

  /* callback argument array */
  cb_arg  *_cbargs;
  unsigned _cbargs_index;

 public:
#ifdef CONFIG_QUEUE_STATS
  struct {
    unsigned long issued;
    unsigned long failed_polls;
    unsigned long list_skips;
    unsigned long polls;
    cpu_time_t total_submit_cycles;
    cpu_time_t max_submit_cycles;
    cpu_time_t total_complete_cycles;
    cpu_time_t max_complete_cycles;
    cpu_time_t total_io_cycles;
    pthread_t pthread;
    unsigned long max_io_size_blocks;    
  } _stats;
#endif

  struct {
    long outstanding;
    uint64_t last_tag;
  } _status __attribute__((aligned(8)));
  
};

#else
#error This is a C++ only header file
#endif

#endif  // __NVME_QUEUE_H__
