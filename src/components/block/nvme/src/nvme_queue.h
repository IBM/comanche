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

#ifndef __NVME_QUEUE_H__
#define __NVME_QUEUE_H__

#if defined(__cplusplus)

#include <pthread.h>
#include <cstdint>
#include <common/cycles.h>
#include <common/types.h>
#include <api/block_itf.h>

#include "nvme_device.h"
#include "nvme_queue.h"
#include "config.h"
#include "types.h"

struct spdk_nvme_qpair;

class Nvme_device;
class Nvme_queue;


/**
 * Represents an IO queue pair on a device
 *
 */
class Nvme_queue
{
  friend class NVMe_storage_device;
  friend class Block_device_component;

private:
  static const bool option_DEBUG = false;
  
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
   * Submit asynchronous IO operation (internal use only)
   * 
   * @param desc IO descriptor
   */
  void submit_async_op_internal(IO_descriptor * desc);

  bool check_completion(uint64_t gwid);
  
  /** 
   * Get the next completion tag.
   * 
   * 
   * @return Async operation tag or 0 if none available.
   */
  //  uint64_t get_next_async_completion();
  
  /** 
   * Get tag of last known completion
   * 
   * @return ID of last completed operation
   */
  //  uint64_t get_last_completion();

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
  inline Nvme_device *device() { return _device;  }
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

  /* we use this list to track pending IO. this FIFO queue
     will only be accessed by the IO thread. we can access
     the last member of the list (which will be the lowest)
     to find the lowest pending tag which tells us the highest
     completed tag! */
  IO_descriptor * _pending_hd = nullptr; 
  IO_descriptor * _pending_tl = nullptr;

  std::vector<std::pair<std::function<void(void*)>, void *> > _shared_work_functions;
  
public:

  void execute_shared_work() {
    for(auto& w: _shared_work_functions) {
      w.first(w.second); /* call registered shared work function */
    }
  }
    
  inline void attach_work(std::function<void(void*)> work_function, void * arg) {
    _shared_work_functions.push_back({work_function,arg});
  }

  inline bool pending_remain() {
    return _pending_hd == nullptr;
  }
  
  /** 
   * In-place removal
   * 
   */
  inline void remove_pending_fifo(IO_descriptor * desc) {
    assert(desc);

    if(option_DEBUG)
      PLOG("removing from  FIFO: %p (tag=%lu)", desc, desc->tag);

    /* written with [A]<->[B]<->[C] removing B */
    IO_descriptor * b_prev = desc->prev;
    IO_descriptor * b_next = desc->next;

    if(b_prev)
      b_prev->next = b_next;
    else
      _pending_hd = b_next;

    if(b_next)
      b_next->prev = b_prev;
    else
      _pending_tl = b_prev;
  }

  /** 
   * Push to front of pending FIFO
   * 
   */
  inline void push_pending_fifo(IO_descriptor * desc) {

    if(option_DEBUG)
      PLOG("pushing FIFO: %p (tag=%lu)", desc, desc->tag);
    
    IO_descriptor * oldnext = nullptr;
    if(_pending_hd)
      oldnext = _pending_hd;
    
    _pending_hd = desc;
    desc->next = oldnext;
    if(oldnext)
      oldnext->prev = desc;
    desc->prev = nullptr;

    /* update tail */
    if(desc->next == nullptr)
      _pending_tl = desc;
  }

  uint64_t get_last_completion() {
    // for the moment go through list
    IO_descriptor * iod = _pending_tl; 
    if(iod == nullptr) return UINT64_MAX;
    // while(iod->next)
    //   iod = iod->next;
    return (iod->tag - 1); /* we know last tag processed is this minus one */
  }

private:
  const float _rdtsc_freq_mhz;
public:
  
#ifdef CONFIG_QUEUE_STATS
  struct {
    unsigned long issued;
    unsigned long lba_count = 0;
    unsigned long failed_polls;
    unsigned long list_skips;
    unsigned long polls;
    cpu_time_t last_report_timestamp = 0;
    cpu_time_t total_submit_cycles;
    cpu_time_t max_submit_cycles;
    cpu_time_t total_complete_cycles;
    cpu_time_t max_complete_cycles;
    cpu_time_t total_io_cycles;
    pthread_t pthread;
    unsigned long max_io_size_blocks;    
  } _stats;
#endif

};

#else
#error This is a C++ only header file
#endif

#endif  // __NVME_QUEUE_H__
