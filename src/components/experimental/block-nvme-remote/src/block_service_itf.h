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

#ifndef __COMANCHE_BLOCK_ITF_H__
#define __COMANCHE_BLOCK_ITF_H__

#include <common/mpmc_bounded_queue.h>

#include "storage_device.h"
#include "volume_agent_session.h"
#include "types.h"
#include "policy.h"

namespace comanche
{

class Storage_device;
class Volume_agent_session;
class Nvme_queue;


class IBlock_service_session
{
public:
    
  /** 
   * Allocate a memory region that can be used for IO
   * 
   * @param size Size of memory in bytes
   * @param alignment Alignment
   * 
   * @return Handle to IO memory region
   */
  virtual void io_memory_t allocate_io_buffer(size_t size, size_t alignment) = 0;

  /** 
   * Free a previously allocated buffer
   * 
   * @param io_mem Handle to IO memory allocated by allocate_io_buffer
   * 
   * @return S_OK on success
   */
  virtual status_t free_io_buffer(io_memory_t io_mem) = 0;

  /** 
   * Get the buffer region pointer from io memory handle
   * 
   * @param io_mem 
   * 
   * @return 
   */
  virtual void * addr(io_memory_t io_mem) = 0;

  /** 
   * Register memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual void register_memory_for_io(void * vaddr, size_t len) = 0;

  /** 
   * Unregister memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  virtual void unregister_memory_for_io(void * vaddr, size_t len) = 0;
  
  /** 
   * Submit an IO operation for asynchronous execution. This method
   * is thread-safe and is expected to be used across multiple threads
   * 
   * @param op COMANCHE_OP_WRITE || COMANCHE_OP_READ
   * @param buffer IO buffer
   * @param lba Logical block address
   * @param lba_count Logical block count
   * 
   * @return tag Work id
   */
  virtual uint64_t async_submit(int op, io_memory_t buffer, uint64_t lba, uint64_t lba_count) = 0;

  /** 
   * Check for completion of a work request. This API is thread-safe.
   * 
   * @param gwid Work request identifier
   * 
   * @return True if completed.
   */
  virtual bool check_completion(uint64_t gwid) = 0;

};


} // comanche

#endif // __COMANCHE_BLOCK_ITF_H__
