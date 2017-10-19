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

#ifndef __COMANCHE_VOLUME_AGENT_SESSION_H__
#define __COMANCHE_VOLUME_AGENT_SESSION_H__

#include "types.h"
#include "channel.h"
#include "agent.h"

class Volume_agent_session
{
private:
  static constexpr bool option_DEBUG = false;

  static constexpr unsigned BUFFER_POOL_SIZE = 256;
  static constexpr unsigned BUFFER_POOL_REGION_SIZE = 4096*2;
  static constexpr unsigned RECV_BUFFER_NUM = BUFFER_POOL_SIZE / 2;

public:
  /** 
   * Constructor
   * 
   * @param core 
   */
  Volume_agent_session(Agent& agent, const char * peer_name, unsigned core);

  ~Volume_agent_session();
  
  /** 
   * Register memory with DMA subsystem
   * 
   * @param addr Region address
   * @param len Length of region in bytes
   * 
   * @return Handle to memory (for use with submit_sync_op)
   */  
  inline channel_memory_t register_region(void * addr, size_t len) {
    return static_cast<channel_memory_t>(_channel.register_memory(addr, len));
  }

  /** 
   * Release a previously register handle
   * 
   * @param region 
   */
  inline void release_region(channel_memory_t region) {
    _channel.release_handle(region);
  }

  /** 
   * Allocate RTE memory and register for RDMA
   * 
   * @param len 
   * @param alignment 
   * 
   * @return 
   */
  channel_memory_t alloc_region(size_t len, size_t alignment = 64);
  
  /** 
   * Free allocated RTE region
   * 
   * @param handle 
   */
  void free_region(channel_memory_t handle);

  /** 
   * Submit a synchronous IO operation to the storage system
   * 
   * @param region Memory region
   * @param lba Logical block address
   * @param lba_count Logical block count
   * @param op IO operation
   * 
   * @return S_OK on success
   */
  status_t submit_sync_op(channel_memory_t region,
                          size_t lba,
                          size_t lba_count,
                          int op);

  /** 
   * Asynchronous IO operation
   * 
   * @param region Memory region
   * @param lba Logical block address
   * @param lba_count Logical block count
   * @param op IO operation
   * 
   * @return Global work ID (gwid)
   */
  uint64_t submit_async(channel_memory_t region,
                        size_t lba,
                        size_t lba_count,
                        int op);

  /** 
   * Check for completion of a specific gwid
   * 
   * @param gwid Global work ID
   * 
   * @return True if gwid complete. False otherwise
   */
  bool check_completion(uint64_t gwid);

  /** 
   * Return gwid of last completion
   * 
   * 
   * @return Last completion gwid
   */
  uint64_t last_completion();
  
  /** 
   * Return last completed gwid
   * 
   * 
   */
  inline uint64_t last_gwid() const { return _completed_wid.load(); }
  
  /** 
   * Poll completions and return outstanding count
   * 
   * 
   * @return 
   */
  unsigned poll_outstanding();
  

private:
  std::atomic<unsigned>  _outstanding_requests; /**< number of outstanding remote requests */
  std::atomic<uint64_t>  _completed_wid;
  Channel                _channel;

  std::unique_ptr<Buffer_manager> _buffer_pool; /**< receive buffer pool */
  
};



#endif // __COMANCHE_VOLUME_AGENT_SESSION_H__
