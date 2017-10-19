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

#include "buffer_manager.h"
#include "channel.h"

Buffer_manager::Buffer_manager(const char * name,
                               Channel& channel,
                               unsigned n,
                               unsigned size)
  : _channel(channel)
#ifndef USE_STD_VECTOR
  ,_mr_ring(name, size)    
#endif
  ,_size(size)
{
  for(unsigned i=0;i<n;i++) {
#ifdef USE_STD_VECTOR    
    _mr_list.push_back(channel.allocate_dpdk_mr(size, -1));
#else
    _mr_ring.sp_enqueue(channel.allocate_dpdk_mr(size, -1));    
#endif
  }
  _free = n;
}

Buffer_manager::~Buffer_manager() {
  PLOG("deleting Buffer_manager %p", this);
#ifdef USE_STD_VECTOR
  for(auto& mr: _mr_list) {
    _channel.release_handle(mr);
  }
#else
  while(!_mr_ring.empty()) {
    channel_memory_t mrp = nullptr;
    _mr_ring.sc_dequeue(mrp);
    mrp->length = _size;
    assert(mrp);
    _channel.release_handle(mrp);
  }      
#endif
  
}

channel_memory_t Buffer_manager::alloc() {    
  if(_free==0) return NULL;
  channel_memory_t mrp = nullptr;
#ifdef USE_STD_VECTOR
  mrp = _mr_list.back();
  _mr_list.pop_back();
#else
  _mr_ring.sc_dequeue(mrp);
#endif
  assert(mrp);
  _free--;
  return mrp;
}

void Buffer_manager::free(channel_memory_t mr) {
#ifdef USE_STD_VECTOR
  _mr_list.push_back(mr);
#else
  _mr_ring.sp_enqueue(mr);
#endif
  _free++;
}

