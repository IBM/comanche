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

#include <rte_malloc.h>
#include "volume_agent_session.h"

Volume_agent_session::Volume_agent_session(Agent& agent,
                                           const char * peer_name,
                                           unsigned core) :
  _outstanding_requests(0),
  _completed_wid(0),
  _channel(false)
{

  /* connect to peer through TCP/IP and establish data plane channel */
  if(agent.connect_peer(peer_name, _channel)!=S_OK) {
    throw Constructor_exception("unable to connect to peer (%s)", peer_name);
  }


  char tmpname[64];
  sprintf(tmpname,"va-buff-mgr-%u",core);

  /* create buffer pool */
  _buffer_pool = std::unique_ptr<Buffer_manager>
    (new Buffer_manager(tmpname,
                        _channel,
                        BUFFER_POOL_SIZE,
                        BUFFER_POOL_REGION_SIZE));

  /* set up recv buffers */
  for(unsigned i=0;i<RECV_BUFFER_NUM;i++) {
    struct ibv_mr * mr = _buffer_pool->alloc();

    struct IO_command * cmd = static_cast<struct IO_command*>(mr->addr);
    cmd->op_flags = COMANCHE_PROTOCOL_FLAG_REQUEST;
    _channel.post_recv((uint64_t)mr,mr);
  }

}

Volume_agent_session::~Volume_agent_session()
{
}

channel_memory_t Volume_agent_session::alloc_region(size_t len, size_t alignment)
{
  void * buff = rte_malloc(NULL, len, alignment);
  assert(buff);
  return register_region(buff, len);
}

void Volume_agent_session::free_region(channel_memory_t handle)
{
  assert(handle);
  rte_free(handle->addr);
}

uint64_t Volume_agent_session::submit_async(channel_memory_t mr_extra,
                                            size_t lba,
                                            size_t lba_count,
                                            int op)
{
  assert(mr_extra->length < MB(4)); /* sanity check */
  
  if(option_DEBUG)
    PMAJOR("submit_sync_op: lba=%ld lba_count=%ld op=%d",
          lba, lba_count, op);

  
  struct ibv_mr * mr0 = nullptr;
  while(_outstanding_requests >= 32) {
    /* we're out of buffers, try to free some up by processing completions */
    poll_outstanding();
  }
  mr0 = _buffer_pool->alloc();
  
  struct IO_command * cmd = static_cast<struct IO_command*>(mr0->addr);
  
  assert(cmd);
  assert(lba_count > 0);

  static uint64_t seq = 1;
  
  /* set up header */
  cmd->magic = COMANCHE_PROTOCOL_MAGIC;
  cmd->op_flags = op | COMANCHE_PROTOCOL_FLAG_REQUEST;
  cmd->lba = lba;
  cmd->lba_count = lba_count;
  cmd->gwid = seq;
  cmd->len = sizeof(struct IO_command) + KB(4);
  seq++;
  mr0->length = sizeof(struct IO_command);
  
  if(option_DEBUG)
    PLOG("posting gwid=%lu", cmd->gwid);
  
  /* asynchronously post send */
  try {
    _channel.post_send((uint64_t) mr0, mr0, mr_extra); /* 2 segment payload */
  }
  catch(...) {
    throw General_exception("volume agent channel post_send failed unexpectedly.");
  }

  _outstanding_requests++;

  return cmd->gwid;
}

status_t Volume_agent_session::submit_sync_op(channel_memory_t region,
                                      size_t lba,
                                      size_t lba_count,
                                      int op)
{
  uint64_t gwid = this->submit_async(region, lba, lba_count, op);

  int retries=0;
  while(!check_completion(gwid)) {
    retries++;
    if(retries > 10000) {
      PWRN("submit_sync_op: timed out on completion");
      return E_NO_RESPONSE;
    }
  }
  return S_OK;
}


bool Volume_agent_session::check_completion(uint64_t gwid)
{
  poll_outstanding(); /* do poll */

  return (_completed_wid.load() >= gwid);
}

uint64_t Volume_agent_session::last_completion()
{
  return _completed_wid.load();
}

unsigned Volume_agent_session::poll_outstanding()
{
  _channel.poll_completions([=](uint64_t wid) {
      struct ibv_mr* mr0 = reinterpret_cast<struct ibv_mr*>(wid);
      struct IO_command * cmd = static_cast<struct IO_command*>(mr0->addr);

      auto comp = _completed_wid.load();
      
      if(cmd->gwid > comp )
        _completed_wid.store(cmd->gwid);
      
      if(cmd->op_flags & COMANCHE_PROTOCOL_FLAG_RESPONSE) {
        /* todo check completion details */
        _outstanding_requests--;
        try {
          _channel.post_recv((uint64_t)mr0,mr0); /* repost for receive */
        }
        catch(...) {
          throw General_exception("repost recv failed in volume_agent");
        }
      }
      else {
        /* send completion */
        _buffer_pool->free(mr0);
      }
      
    });
  
  return _outstanding_requests;
}


