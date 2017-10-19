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

#ifndef __COMANCHE_POLICY_H__
#define __COMANCHE_POLICY_H__

#include <api/block_itf.h>
#include <memory>
#include "types.h"

/** 
 * A policy defines how data is distributed between local
 * and remote storage.  This aims to be a pluggable design
 * so that new policies cann easily be integrated.
 * 
 */
class Policy
{
public:
  virtual ~Policy() {}
  virtual uint64_t issue_op(struct IO_command* cmd) = 0;
  virtual uint64_t gratuitous_completion() = 0;
  virtual bool check_completion(uint64_t gwid) = 0;
  virtual io_memory_t allocate_io_buffer(size_t size, size_t alignment, int numa_node) = 0;
  virtual status_t reallocate_io_buffer(io_memory_t io_mem, size_t size, unsigned alignment) = 0;
  virtual status_t free_io_buffer(io_memory_t io_mem) = 0;
  virtual void * get_addr(io_memory_t io_mem) = 0;
  virtual uint64_t get_phys_addr(io_memory_t io_mem) = 0;
  virtual io_memory_t register_io_buffer(void * ptr, size_t size) = 0;
  virtual status_t unregister_io_buffer(void * ptr, size_t size) = 0;

  /** 
   * Get logical volume information
   * 
   * @param devinfo device information structure
   * 
   */
  virtual void get_volume_info(Component::VOLUME_INFO& devinfo) = 0;

};


#endif //__COMANCHE_POLICY_H__
