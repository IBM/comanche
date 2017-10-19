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

#include <common/cpu.h>
#include "storage_agent_session.h"
#include "operations.h"

#undef USE_NULL_DEVICE /**< null device for testing */

void
Storage_agent_session::
session_thread(int port, unsigned core)
{
  static constexpr unsigned BUFFER_POOL_SIZE = 256;
  static constexpr unsigned BUFFER_POOL_REGION_SIZE = 4096*2;
  static constexpr unsigned RECV_BUFFER_NUM = BUFFER_POOL_SIZE;

  {
    char tmpname[32];
    sprintf(tmpname,"sa-session-%u",core);
    pthread_setname_np(pthread_self(), tmpname);
  }

  /* set thread affinity */
  {
    cpu_mask_t mask;
    mask.set_bit(core);
    set_cpu_affinity_mask(mask);
  }

  Channel channel(Channel::FLAGS_USE_DPDK);
  channel.wait_for_connect(nullptr, /* device name */ port);

  
  /* create buffer pool - we actually don't use it as a pool here. */
  char tmpnam[32];
  sprintf(tmpnam,"sa-buff-mgr-%d",port);
  auto buffer_pool = new Buffer_manager(tmpnam,
                                        channel,
                                        BUFFER_POOL_SIZE,
                                        BUFFER_POOL_REGION_SIZE);

  /* set up recv buffers */
  for(unsigned i=0;i<RECV_BUFFER_NUM;i++) {
    struct ibv_mr * mr = buffer_pool->alloc();
    struct IO_command * cmd = static_cast<struct IO_command*>(mr->addr);
    cmd->op_flags = COMANCHE_PROTOCOL_FLAG_RESPONSE;
    assert(mr->length == BUFFER_POOL_REGION_SIZE);
    channel.post_recv((uint64_t)mr,mr); /* munge pointer to mr into wid */
  }

  PLOG("Entering channel poll loop...");


  /* data operations */
#ifdef USE_NULL_DEVICE
  Null_IO_operation data_op;
#else
  Nvme_IO_operation data_op(_queue);
#endif
    
  uint64_t completed_wid = 0;
  while(!_exit) {

    /*--- process channel-received work items */
    
    channel.poll_completions([&data_op,&channel,&completed_wid,&buffer_pool](uint64_t wid) {

        channel_memory_t mr0 = reinterpret_cast<channel_memory_t>(wid);
        struct IO_command * cmd = static_cast<struct IO_command*>(mr0->addr);
        if(cmd->gwid > completed_wid) completed_wid = cmd->gwid;
        if(unlikely(completed_wid == ULLONG_MAX)) completed_wid=0; /* might be wrap around issue */
        
        if(cmd->op_flags & COMANCHE_PROTOCOL_FLAG_REQUEST) {

          /* submit for data operation */
          data_op.submit(mr0);
        }
        else {
          /* response completion */

          /* re-post for next recv */
          mr0->length = BUFFER_POOL_REGION_SIZE;
          channel.post_recv((uint64_t) mr0, mr0);
        }
        
      });

    /* process data operation completions */
    {
      channel_memory_t mr;
      while ((mr = data_op.get_next_completion()) != nullptr) {

        struct IO_command * cmd = reinterpret_cast<struct IO_command *>(mr->addr);
        /* complete, post response */
        if (option_DEBUG) {
          PINF("data op complete ; posting response :%lu", cmd->gwid);
        }
        
        /* send response (IO_command only) */
        cmd->op_flags = COMANCHE_PROTOCOL_FLAG_RESPONSE;
        mr->length = sizeof(struct IO_command);
        channel.post_send((uint64_t) mr, mr);
      }
    }

    
  }

      // auto index = wid - 1;
      
      // /* a low wid is a recv completion; a high wid is a response completion */
      // if(index < MAX_OUTSTANDING_RECVS) {

      //   if(option_DEBUG) {
      //     struct IO_command * cmd = static_cast<struct IO_command *>(recv_mb[wid-1]->addr);
      //     PLOG("recv event: wid=%lu (gwid=%lu)", wid, cmd->gwid);
      //   }
	
      //   auto index = wid - 1;	  
      //   auto mr = recv_mb[index];

      //   /* submit for data data processing */
      //   data_op.submit(mr,wid);	  
      // }
      // else {
      //   auto index = wid - 1 - MAX_OUTSTANDING_RECVS;

      //   if(option_DEBUG) {
      //     struct IO_command * cmd = static_cast<struct IO_command *>(recv_mb[index]->addr);
      //     PLOG("response completed: %lu (gwid=%lu)", (wid-MAX_OUTSTANDING_RECVS), cmd->gwid);
      //   }
	  
      //   /* repost as recv completion */
      //   channel.post_recv_wid(recv_mb[index],index+1);
      // }

  
  /* clean up */
  delete buffer_pool;
}


Storage_agent_session::
Storage_agent_session(Storage_agent * parent,
                      const char * peer_name,
                      int port,
                      Storage_device* storage_device,
                      unsigned core) :
  _peer_name(peer_name),
  _sa(parent),
  _storage_device(storage_device),
  _exit(false),
  _core(core)
{
  assert(parent);
  assert(storage_device);

  PLOG("New session: device (size=%ld GB)", REDUCE_GB(_storage_device->nvme_device()->get_size_in_bytes(1)));
  /* create queue */
  _queue = _storage_device->allocate_queue();
  assert(_queue);  

  /* create thread for session */
  _thread = new std::thread(&Storage_agent_session::session_thread, this, port, core);
}

Storage_agent_session::
~Storage_agent_session() {
  TRACE();

  _exit = true;
  _thread->join();
  delete _thread;

  _storage_device->free_queue(_queue);
}


