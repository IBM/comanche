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

#ifndef __COMANCHE_POLICY_MIRROR_H__
#define __COMANCHE_POLICY_MIRROR_H__


#include <memory>
#include "types.h"
#include "policy.h"

class Storage_device;
class Volume_agent_session;
class Nvme_queue;

/** 
 * Basic policy for local and remote mirroring
 * 
 */
class Policy_mirror : public Policy
{
public:
  Policy_mirror(Storage_device * storage_device,
                Volume_agent_session * remote_volume);

  virtual ~Policy_mirror();
  
  /** 
   * Called for each IO issue
   * 
   * @param cmd Command 
   * 
   * @return Global work identifier
   */
  uint64_t issue_op(struct IO_command* cmd);

  /** 
   * Called to perform gratuitous completions
   * 
   * 
   * @return Last gwid processed
   */
  uint64_t gratuitous_completion();

  /** 
   * Check for completion of a specific gwid
   * 
   * @param gwid Global work identifier from issue_op
   * 
   * @return true if complete
   */
  bool check_completion(uint64_t gwid);

  /** 
   * Allocate IO memory.  Often the memory needs to
   * be attached to a remote IO channel etc.
   * 
   * @param size Size of buffer in bytes
   * @param alignment Alignment in bytes
   * @param numa_node NUMA node designator
   * 
   * @return Handle to IO memory
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
  status_t reallocate_io_buffer(io_memory_t io_mem, size_t size, unsigned alignment);

  /** 
   * Free previously allocated IO memory.
   * 
   * @param io_mem Memory handle
   * 
   * @return S_OK on success
   */
  status_t free_io_buffer(io_memory_t io_mem);

  /** 
   * Get the virtual address of allocated IO memory
   * 
   * @param io_mem 
   * 
   * @return 
   */
  inline void * get_addr(io_memory_t io_mem)
  {
    auto mr = reinterpret_cast<channel_memory_t>(io_mem);  
    return mr->addr;    
  }

  /** 
   * Get physical address for memory
   * 
   * @param io_mem IO memory handle
   * 
   * @return physical address
   */
  uint64_t get_phys_addr(io_memory_t io_mem);


  /** 
   * Register memory (not allocated through the policy) to be used
   * 
   * @param ptr Pointer to memory
   * @param size Size in bytes
   * 
   * @return IO memory handle
   */
  io_memory_t register_io_buffer(void * ptr, size_t size);

  /** 
   * Unregister a buffer for IO with DMA
   * 
   * @param ptr 
   * @param size 
   * 
   * @return 
   */
  status_t unregister_io_buffer(void * ptr, size_t size);

  /** 
   * Get device information
   * 
   * @param devinfo device information structure
   * 
   */
  void get_volume_info(Component::VOLUME_INFO& devinfo);
  
private:
  Nvme_queue *           _local_volume;
  Volume_agent_session * _remote_volume;
  Storage_device *       _device;
};


#endif //__COMANCHE_POLICY_H__
