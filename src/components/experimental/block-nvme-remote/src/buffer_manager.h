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

#ifndef __COMANCHE_BUFFER_MANAGER_H__
#define __COMANCHE_BUFFER_MANAGER_H__

#include <vector>
#include <infiniband/verbs.h>
#include "types.h"
#include "channel.h"

//#ifdef USE_STD_VECTOR /**< std::list version seems slower than DPDK ring */

class Channel;

/** 
 * Single threaded channel buffer (memory region) management class. 
 * 
 */
class Buffer_manager
{
public:
  /** 
   * Constructor
   * 
   * @param name Name of instance
   * @param channel Channel to allocate MRs from
   * @param n Number of MRs to allocate
   * @param size Size of each MR
   */
  Buffer_manager(const char * name, Channel& channel, unsigned n, unsigned size);

  /** 
   * Destructor
   * 
   */
  ~Buffer_manager();

  /** 
   * Alloc an MR (memory region)
   * 
   * 
   * @return Handle to memory region. Must be freed with Buffer_manager::free
   */
  channel_memory_t alloc();

  /** 
   * Free previously allocated MR
   * 
   * @param mr MR to free.
   */
  void free(channel_memory_t mr);

  /** 
   * Check for empty status
   * 
   * 
   * @return True if no more MRs available.
   */
  inline bool empty() const { return _free == 0; }


private:
  Channel&                            _channel;
#ifdef USE_STD_VECTOR
  std::vector<channel_memory_t>       _mr_list;
#else
  DPDK::Ring_buffer<channel_memory_t> _mr_ring;
#endif
  int                                 _free;
  size_t                              _size;
};



#endif // __COMANCHE_BUFFER_MANAGER_H__


