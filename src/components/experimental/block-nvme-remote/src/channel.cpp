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

#include <rte_common.h>
#include <rte_mempool.h>
#include <rte_malloc.h>
#include <rte_errno.h>
#include <spdk/env.h>
#include "channel.h"
#include "eal_init.h"

Channel::Channel(int core, int channel_flags) :
  _recv_mempool(NULL),
  _use_dpdk(channel_flags & FLAGS_USE_DPDK),
  _abort(false),
  _core(core)
{
  assert(core >= 0);
}
  
Channel::~Channel() {
  TRACE();
  /* clean up memool */
  if(_recv_mempool) {
    struct ibv_mr * mr;
    while(rte_ring_mc_dequeue(_recv_mempool, (void**) &mr)==0) {
      if(_use_dpdk)
        rte_free(mr->addr);
      else
        ::free(mr->addr);
    }
    rte_ring_free(_recv_mempool);
  }
}

status_t Channel::connect(const char * server_name,
                          const char * device_name,
                          int port) {
  if(_transport.connect(server_name,device_name,port)!=S_OK)
    throw General_exception("RDMA transport connect failure");

  if(_use_dpdk)
    init_dpdk_memory(port);
  else
    init_memory(port);

  return S_OK;
}

status_t Channel::wait_for_connect(const char * device_name,
                                   int port)
{
  return connect(NULL, device_name, port);
}


// Channel::memory_handle_t Channel::post_recv(int * completion)
// {
//   assert(_recv_mempool);
//   struct ibv_mr * mr;
//   if(rte_ring_mc_dequeue(_recv_mempool, (void**) &mr)!=0)
//     throw General_exception("no more memory regions in channel recv_pool");
//   assert(mr);

//   _transport.post_recv(completion, mr);

//   return static_cast<memory_handle_t>(mr);
// }



// Channel::memory_handle_t Channel::dpdk_recv()
// {
//   assert(_recv_mempool);
//   struct ibv_mr * mr;
//   if(rte_ring_mc_dequeue(_recv_mempool, (void**) &mr)!=0)
//     throw General_exception("no more memory regions in channel recv_pool");
//   assert(mr);

//   int completion = 0;
//   _transport.post_recv(&completion, mr);

//   // while(!completion && !_abort)
//   //   _transport.poll_completions();
//   //  while(_transport.poll_completions() > 0 && !_abort);
//   assert(0); // broken

//   if(_abort) {
//     assert(0);
//     release_handle(mr);
//     return NULL;
//   }
  
//   return static_cast<memory_handle_t>(mr);
// }

// Channel::memory_handle_t Channel::alloc_handle()
// {
//   struct ibv_mr * mr;
//   if(rte_ring_mc_dequeue(_recv_mempool, (void**) &mr)!=0)
//     throw General_exception("no more memory regions in channel recv_pool");
//   assert(mr);
//   return mr;
// }

void Channel::release_handle(channel_memory_t mr)
{
  rte_ring_mp_enqueue(_recv_mempool, (void*) mr); /* put back on ring */ 
}


struct ibv_mr * Channel::allocate_dpdk_mr(size_t size, int numa_socket)
{
  void * addr = rte_zmalloc_socket("channel-mem",size, 0x1000 /* align */, numa_socket);
  assert(addr);
  return _transport.register_memory(addr, size);
}

struct ibv_mr * Channel::allocate_mr(size_t size, int numa_socket)
{
  void * addr = aligned_alloc(size, 0x1000);
  assert(addr);
  return _transport.register_memory(addr, size);
}


void Channel::init_dpdk_memory(unsigned port)
{
  char tmpname[64];
  sprintf(tmpname,"ch-recv-pool-%u-%d",port,_core);
    
  /* create lockfree ring of pointers to buffers */
  _recv_mempool = rte_ring_create(tmpname,
                                  RECV_BUFFER_COUNT,
                                  -1, /* numa socket */
                                  0); /* flags */
      
  if(_recv_mempool == NULL)
    throw General_exception("Channel recv rte_ring_create failed (%d)", rte_errno);

  /* populate ring  */
  while(!rte_ring_full(_recv_mempool)) {
    struct ibv_mr* mr = allocate_dpdk_mr(RECV_BUFFER_SIZE,-1 /* numa */);
    assert(mr);
    rte_ring_sp_enqueue(_recv_mempool, (void*)mr);
  }
  PLOG("initialized DPDK receive buffers for Channel OK");
}

void Channel::init_memory(unsigned port)
{
  char tmpname[64];
  sprintf(tmpname,"chann-recv-pool-%u-%d",port,_core);
  
  /* create lockfree ring of pointers to buffers */
  _recv_mempool = rte_ring_create(tmpname,
                                  RECV_BUFFER_COUNT,
                                  -1, /* numa socket */
                                  0); /* flags */
      
  if(_recv_mempool == NULL)
    throw General_exception("Channel recv mempool create failed (%d)", rte_errno);

  /* populate ring  */
  while(!rte_ring_full(_recv_mempool)) {
    struct ibv_mr* mr = allocate_mr(RECV_BUFFER_SIZE,-1 /* numa */);
    assert(mr);
    rte_ring_sp_enqueue(_recv_mempool, (void*)mr);
  }
}

