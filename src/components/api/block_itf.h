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

#ifndef __API_BLOCK_ITF__
#define __API_BLOCK_ITF__

#ifdef __cplusplus

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <functional>
#include <cstdint>
#include <mutex>
#include <semaphore.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <common/cpu.h>
#include <component/base.h>
#include "memory_itf.h"

namespace Core
{
class Poller;
}

namespace Component
{

using block_t      = uint64_t;
using status_t     = int;
using workid_t     = uint64_t;
using addr_t       = uint64_t;

static constexpr unsigned VOLUME_INFO_MAX_NAME = 64;

struct VOLUME_INFO;

struct VOLUME_INFO
{
public:
  VOLUME_INFO() {
    __builtin_memset(this, 0, sizeof(VOLUME_INFO));
  }

  char volume_name[VOLUME_INFO_MAX_NAME];  
  unsigned block_size;
  uint64_t hash_id;
  uint64_t max_lba;
  uint64_t max_dma_len;
  unsigned distributed : 1;
  unsigned sw_queue_count : 7; /* counting from 0, i.e. 0 equals 1 queue */
  void dump() {
    PINF("VOLUME_INFO: (%s) %u %lu %lu max_dma=%lu dis=%u swqc=%u",
         volume_name, block_size, hash_id, max_lba, max_dma_len, distributed,
         sw_queue_count);
  }
};


class IBlock_device;

/** 
 * Factory pattern component
 * 
 */
class IBlock_device_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfac00b3c,0x7989,0x11e3,0x8f16,0xbc,0x30,0x5b,0xdc,0x75,0x4d);

  /** 
   * Basic instance creation
   * 
   * @param config Configuration string
   * @param cpuset Optional CPU set for IO threads
   * @param poller Optional poller for shared threading
   * 
   * @return 
   */
  virtual IBlock_device * create(std::string config,
                                 cpu_mask_t * cpuset = nullptr,
                                 Core::Poller * poller = nullptr) = 0;
};


/** 
 * Block device interface
 * 
 */
class IBlock_device : public IZerocopy_memory
{
public:
  DECLARE_INTERFACE_UUID(0xbbbb2b3f,0x7989,0x11e3,0x8f16,0xbc,0x30,0x5b,0xdc,0x75,0x4d);

  typedef void (*io_callback_t)(uint64_t, void*, void*);  

  class Semaphore {
  public:
    Semaphore() { sem_init(&_sem,0,0); }
    inline void post() { sem_post(&_sem); }
    inline void wait() { sem_wait(&_sem); }
  private:
    sem_t _sem;
  };
  
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
  virtual workid_t async_read(io_buffer_t buffer,
                              uint64_t buffer_offset,
                              uint64_t lba,
                              uint64_t lba_count,
                              int queue_id = 0,
                              io_callback_t cb = nullptr,
                              void * cb_arg0 = nullptr,
                              void * cb_arg1 = nullptr) = 0;   
  /** 
   * Synchronous read
   * 
   * @param buffer IO buffer
   * @param offset Offset in logical blocks from the start of the IO buffer
   * @param lba logical block address
   * @param lba_count number of blocks
   * @param queue_id Logical queue identifier (counting from 0, -1=unspecified)
   */
  virtual void read(io_buffer_t buffer,
                    uint64_t buffer_offset,
                    uint64_t lba,
                    uint64_t lba_count,
                    int queue_id = 0) {

#ifdef __clang__
    static thread_local Semaphore sem;
#else
    static __thread Semaphore sem; // GCC
#endif
    
    workid_t wid = async_read(buffer, buffer_offset, lba, lba_count, queue_id,
                              [](uint64_t gwid, void* arg0, void* arg1)
                              {
                                ((Semaphore *)arg0)->post();
                              },
                              (void*) &sem);
    sem.wait();
  }

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
  virtual workid_t async_write(io_buffer_t buffer,
                               uint64_t buffer_offset,
                               uint64_t lba,
                               uint64_t lba_count,
                               int queue_id = 0,
                               io_callback_t cb = nullptr,
                               void * cb_arg0 = nullptr,
                               void * cb_arg1 = nullptr) = 0;
  
  /** 
   * Synchronous write
   * 
   * @param buffer IO buffer
   * @param offset Offset in logical blocks from the start of the IO buffer
   * @param lba logical block address
   * @param lba_count number of blocks
   * @param queue_id Logical queue identifier (counting from 0, -1=unspecified)
   */
  virtual void write(io_buffer_t buffer,
                     uint64_t buffer_offset,
                     uint64_t lba,
                     uint64_t lba_count,
                     int queue_id = 0) {

#ifdef __clang__
    static thread_local Semaphore sem;
#else
    static __thread Semaphore sem; // GCC
#endif
    
    workid_t wid = async_write(buffer, buffer_offset, lba, lba_count, queue_id,
                              [](uint64_t gwid, void* arg0, void* arg1)
                              {
                                ((Semaphore *)arg0)->post();
                              },
                               (void*) &sem);
    sem.wait();
  }

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
  virtual bool check_completion(workid_t gwid, int queue_id = 0) = 0;

  /** 
   * Get device information
   * 
   * @param devinfo pointer to VOLUME_INFO struct
   * 
   * @return S_OK on success
   */
  virtual void get_volume_info(VOLUME_INFO& devinfo) = 0;

  /** 
   * Attach a work function. This will be called by the IO threads in their 
   * main servicing loop.
   * 
   * @param work_function Work function pointer
   * @param arg Argument to pass to work function
   * @param queue_id Logical queue identifier (counting from 0, -1=unspecified)
   */
  virtual void attach_work(std::function<void(void*)> work_function, void * arg, int queue_id = 0) {
    throw API_exception("not implemented.");
  }

  /**
   * Attach a handler for failed async IOPs etc.  Not all components
   * will implement this call.
   * 
   * @param cb Error call back function
   */
  virtual void attach_error_callback(std::function<void(status_t)>) {
    throw API_exception("not implemented.");
  }
};

} //< namespace Component

#endif

#endif 
