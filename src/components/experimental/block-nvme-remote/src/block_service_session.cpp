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
#include "config.h"
#include "block_service_session.h"
#include "nvme_queue.h"
#include "types.h"
#include <common/assert.h>
#include <spdk/env.h>

static bool g_service_thread_ready = false;

Block_service_session::Block_service_session(Policy* policy,
                                             unsigned core) :
  _policy(policy),
  _exit(false)
{
  /* create mpmc sleeping queues, which means that the consumer thread will
     sleep when there is nothing to service */
  _lfq0 = new command_queue_t(LFQ_SIZE,&_std_alloc);

  _service_thread = new std::thread(&Block_service_session::__thread_entry,
                                    this, core);

  while(!g_service_thread_ready) usleep(100);
}

Block_service_session::~Block_service_session() {

  _exit = true;
  _lfq0->exit_threads(); /* because this is a sleeping queue */

  /* ordered clean up */
  _service_thread->join();
  delete _service_thread;
  delete _lfq0;
  // do not delete _policy;
}


/** 
 * Main service thread routine.  Currently -O2 is causing some bug to emerge so
 * we force this routine to -O1
 * 
 */
// __attribute__((optimize("O1"))) 
void Block_service_session::service_thread_entry(unsigned core)
{
  PLOG("block service session: worker thread started (thread=%p)", (void*) pthread_self());
  pthread_setname_np(pthread_self(), "cli-sess-svc-thr");
  set_cpu_affinity(1UL << core);
  
  uint64_t issued_gwid    = 0;
  uint64_t completed_gwid = 0;  
  struct IO_command * cmd = nullptr;

  g_service_thread_ready = true;
  
  while(!_exit) {

    cmd = nullptr;

    /* process a submission if condition is OK */
    if((issued_gwid - completed_gwid) > OUTSTANDING_ISSUE_LIMIT) {
      /* cant't service any more requests */
    }
    else if(_lfq0->empty() == false) { 
     if(!_lfq0->dequeue(cmd))
        break; /* false means the queue was interrupted to exit */
    }

    if(cmd) {
      /* at this point the command gwid has not been assigned. the caller (enqueuer)
         is notified by flipping the COMANCHE_PROTOCOL_FLAG_GWID_VALID bit
      */
      if(cmd->magic != COMANCHE_PROTOCOL_MAGIC)
        throw Program_exception("cmd structure is corrupt from client session LFQ");

      assert(!(cmd->op_flags & COMANCHE_PROTOCOL_FLAG_SIGNAL_CALLER));
      
      if(cmd->op_flags & COMANCHE_PROTOCOL_FLAG_REQUEST) {
        //        PLOG("message: COMANCHE_PROTOCOL_FLAG_REQUEST");
        auto gwid = _policy->issue_op(cmd);
        cmd->gwid = gwid;
        if(cmd->gwid > issued_gwid) issued_gwid = cmd->gwid;
        
        /* this is the signal; do it last */
        cmd->op_flags = COMANCHE_PROTOCOL_FLAG_SIGNAL_CALLER; 
      }
      else if(cmd->op_flags == COMANCHE_PROTOCOL_FLAG_CHECK_COMPLETION) {
        //        PLOG("message: COMANCHE_PROTOCOL_FLAG_CHECK_COMPLETION");
        cmd->gwid = _policy->check_completion(cmd->gwid);
        /* this is the signal; do it last */
        cmd->op_flags = COMANCHE_PROTOCOL_FLAG_SIGNAL_CALLER;
      }
    }
    else {
      /* gratutious completion check ; we have to service completions or the RDMA etc.
         will get upset.  This isn't quite what we want, since if a client doesn't collect
         their response, we are left polling.
      */
      completed_gwid = _policy->gratuitous_completion();
    }    
  }
  
  PINF("service thread (%p) exited:", (void*) pthread_self());
}

void Block_service_session::__thread_entry(Block_service_session * parent, unsigned core)
{
  parent->service_thread_entry(core);
}


uint64_t Block_service_session::async_submit(int op,
                                             io_memory_t buffer,
                                             uint64_t offset,
                                             uint64_t lba,
                                             uint64_t lba_count)
{
#ifdef CONFIG_CHECKS_VALIDATE_POINTERS
  channel_memory_t mr = reinterpret_cast<channel_memory_t>(buffer);
  if(!check_ptr_valid(mr,sizeof(mr))) throw Program_exception("memory corruption");
  assert(mr->length < MB(4)); // early sanity check
#endif
  assert(buffer);
  
  struct IO_command cmd;
  cmd.magic = COMANCHE_PROTOCOL_MAGIC;
  cmd.mrdesc = buffer;
  cmd.offset = offset;
  cmd.lba = lba;
  cmd.lba_count = lba_count;
  cmd.op_flags = op | COMANCHE_PROTOCOL_FLAG_REQUEST;

  __sync_synchronize();
  
  _lfq0->enqueue((struct IO_command *)&cmd);

  /* we might need a back-off strategy */
  uint64_t attempts = 0;
  while(!(cmd.op_flags & COMANCHE_PROTOCOL_FLAG_SIGNAL_CALLER)) {
    cpu_relax();
    attempts++;

    if(attempts > IO_ATTEMPTS) {
       PERR("IO_ATTEMPTS exceeded: op=%d lba=%lx lba_count=%ld", op, lba, lba_count);
       throw General_exception("Block_service_session::async_submit failed");
    }
  }

  return cmd.gwid;
}

bool Block_service_session::check_completion(uint64_t gwid)
{
  struct IO_command cmd;
  cmd.magic = COMANCHE_PROTOCOL_MAGIC;
  cmd.gwid = gwid;
  cmd.op_flags = COMANCHE_PROTOCOL_FLAG_CHECK_COMPLETION;

  __sync_synchronize();
  
  _lfq0->enqueue((struct IO_command *)&cmd);

  /* we might need a back-off strategy */
  uint64_t attempts = 0;
  while(!(cmd.op_flags & COMANCHE_PROTOCOL_FLAG_SIGNAL_CALLER)) {
    cpu_relax();
    attempts++;
    if(attempts > IO_ATTEMPTS)
      throw General_exception("Block_service_session: check_completion live-lock");
  }

  return cmd.gwid;
}


io_memory_t Block_service_session::allocate_io_buffer(size_t size, size_t alignment, int numa_node)
{
  return _policy->allocate_io_buffer(size,alignment,numa_node);
}

status_t Block_service_session::realloc_io_buffer(io_memory_t io_mem, size_t size, unsigned alignment)
{
  return S_OK;//TODO
}

status_t Block_service_session::free_io_buffer(io_memory_t io_mem)
{
  return _policy->free_io_buffer(io_mem);
}

void * Block_service_session::virt_addr(io_memory_t io_mem)
{  
  return _policy->get_addr(io_mem);
}

addr_t Block_service_session::phys_addr(io_memory_t io_mem)
{
  return _policy->get_phys_addr(io_mem);
}  

void Block_service_session::register_memory_for_io(void * vaddr, size_t len)
{
  _policy->register_io_buffer(vaddr,len);
}

void Block_service_session::unregister_memory_for_io(void * vaddr, size_t len)
{
  _policy->unregister_io_buffer(vaddr,len);
}

