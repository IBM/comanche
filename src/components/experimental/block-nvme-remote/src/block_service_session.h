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

#ifndef __COMANCHE_BLOCK_SVC_H__
#define __COMANCHE_BLOCK_SVC_H__

#include <common/mpmc_bounded_queue.h>

#include "storage_device.h"
#include "volume_agent_session.h"
#include "types.h"
#include "policy.h"


class Storage_device;
class Volume_agent_session;
class Nvme_queue;


class Block_service_session
{
private:
  static constexpr size_t LFQ_SIZE = 32;
  static constexpr size_t OUTSTANDING_ISSUE_LIMIT = 2048;
  static constexpr size_t IO_ATTEMPTS = 1000000000ULL;
  
public:
  /** 
   * Constructor
   * 
   * @param storage_device 
   * @param remote_volume 
   * @param core 
   */
  Block_service_session(Policy* policy,
                        unsigned core);

  /** 
   * Destructor
   * 
   */
  ~Block_service_session();
  
  /** 
   * Allocate a memory region that can be used for IO
   * 
   * @param size Size of memory in bytes
   * @param alignment Alignment
   * @param numa_node NUMA node or -1 for any
   * 
   * @return Handle to IO memory region
   */
  io_memory_t allocate_io_buffer(size_t size, size_t alignment, int numa_node);

  /** 
   * Re-allocate area of memory
   * 
   * @param io_mem Memory handle (from allocate_io_buffer)
   * @param size New size of memory in bytes
   * @param alignment Alignment in bytes
   * 
   * @return S_OK or E_NO_MEM (unable) or E_NOT_IMPL
   */
  status_t realloc_io_buffer(io_memory_t io_mem,
                             size_t size,
                             unsigned alignment);
  

  /** 
   * Free a previously allocated buffer
   * 
   * @param io_mem Handle to IO memory allocated by allocate_io_buffer
   * 
   * @return S_OK on success
   */
  status_t free_io_buffer(io_memory_t io_mem);

  /** 
   * Get the buffer region pointer from IO memory handle
   * 
   * @param io_mem IO memory handle
   * 
   * @return virtual address (pointer)
   */
  void * virt_addr(io_memory_t io_mem);

  /** 
   * Get the physical address from an IO memory handle
   * 
   * @param io_mem IO memory handle
   * 
   * @return physical address
   */
  addr_t phys_addr(io_memory_t io_mem);
  
  /** 
   * Register memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  void register_memory_for_io(void * vaddr, size_t len);

  /** 
   * Unregister memory for DMA with the SPDK subsystem.
   * 
   * @param vaddr 
   * @param len 
   */
  void unregister_memory_for_io(void * vaddr, size_t len);
  
  /** 
   * Submit an IO operation for asynchronous execution. This method
   * is thread-safe and is expected to be used across multiple threads
   * 
   * @param op COMANCHE_OP_WRITE || COMANCHE_OP_READ
   * @param buffer IO buffer
   * @param buffer_offset Offset in blocks
   * @param lba Logical block address
   * @param lba_count Logical block count
   * 
   * @return tag Work id
   */
  uint64_t async_submit(int op,
                        io_memory_t buffer,
                        uint64_t buffer_offset,
                        uint64_t lba,
                        uint64_t lba_count);

  /** 
   * Check for completion of a work request. This API is thread-safe.
   * 
   * @param gwid Work request identifier
   * 
   * @return True if completed.
   */
  bool check_completion(uint64_t gwid);

  /** 
   * Get device information
   * 
   * @param devinfo 
   * 
   * @return 
   */
  inline void get_volume_info(Component::VOLUME_INFO& devinfo) {
    _policy->get_volume_info(devinfo);
  }

private:

  void service_thread_entry(unsigned core);
  static void __thread_entry(Block_service_session * parent, unsigned core);

  Policy *                                               _policy;
  std::thread *                                          _service_thread;
  bool                                                   _exit;
  Common::Std_allocator                                  _std_alloc;

  //  typedef Common::Mpmc_bounded_lfq_sleeping<struct IO_command*> command_queue_t;
  typedef Common::Mpmc_bounded_lfq<struct IO_command*> command_queue_t;
  command_queue_t* _lfq0; /**< fifo from client to service thread */

};




#endif // __COMANCHE_BLOCK_SVC_H__
