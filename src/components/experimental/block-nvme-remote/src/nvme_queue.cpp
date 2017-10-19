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

#include <common/utils.h>
#include <common/exceptions.h>
#include <common/cycles.h>

#include <pthread.h>
#include <assert.h>
#include <rte_ring.h>
#include <rte_errno.h>
#include <spdk/nvme.h>

#include "nvme_queue.h"
#include "nvme_device.h"

Nvme_queue::Nvme_queue(Nvme_device* device,
                       unsigned qid,
                       struct spdk_nvme_qpair* const qpair)
  :
  _device(device),
  _qpair(qpair),
  _completion_list("nvmeq",qid,NUM_SUB_FRAMES)
{
  if(option_DEBUG)
    PLOG("New Nvme_queue: %u", qid);
  
  assert(device);
  assert(qpair);

  _status.outstanding = 0;
  _status.last_tag = 0;
  
  _ns_id = spdk_nvme_ns_get_id(device->ns());
  _queue_id = qid;
  _block_size = _device->get_block_size(1); // namespace id
  _max_lba = spdk_nvme_ns_get_num_sectors(device->ns());
  
  /* create callback argument space */
  _cbargs_index = 0;
  _cbargs = (cb_arg *) malloc(sizeof(cb_arg) * NUM_SUB_FRAMES);
  memset(_cbargs, 0, sizeof(cb_arg) * NUM_SUB_FRAMES);
  
#ifdef CONFIG_QUEUE_STATS
  /* reset stats */
  _stats.issued = 0;
  _stats.polls = 0;
  _stats.failed_polls = 0;
  _stats.list_skips = 0;
  _stats.pthread = 0;
  _stats.total_submit_cycles = 0;
  _stats.max_submit_cycles = 0;
  _stats.total_complete_cycles = 0;
  _stats.max_complete_cycles = 0;
  _stats.max_io_size_blocks = 0;
  _stats.total_io_cycles = 0;
#endif
  
  PLOG("created new IO queue: namespace=%d max_lba=%ld", _ns_id, _max_lba);
}

Nvme_queue::~Nvme_queue()
{
  assert(_qpair);

  PLOG("freeing nvme io queue pair: %p", _qpair);
  int rc = spdk_nvme_ctrlr_free_io_qpair(_qpair);
  if (rc)
    throw General_exception("spdk_nvme_ctrlr_free_io_qpair failed unexpectedly.");

  free(_cbargs);
}

/* --------- sync operations ----------*/

static void
sync_io_complete(void* arg, const struct spdk_nvme_cpl* completion)
{
  *((int*)arg) = TRANSMIT_STATUS_COMPLETE;
}

status_t
Nvme_queue::submit_sync_op(void* buffer,
                           uint64_t lba,
                           uint64_t lba_count,
                           int op)
{
  assert(buffer);
  assert(check_aligned(buffer, CONFIG_IO_MEMORY_ALIGNMENT_REQUIREMENT));

  if (option_DEBUG)
    PLOG("NVMe IO op (op=%d, lba=%ld, count=%ld)", op, lba, lba_count);

  // bounds check LBA
  if (lba + lba_count >= _max_lba)
    throw API_exception("lba out of bounds");

  volatile int status = TRANSMIT_STATUS_WRITE_INFLIGHT;

  int rc = 0;

  wmb();
  
  if (op == OP_FLAG_READ) {
    if (option_DEBUG)
      PLOG("spdk_nvme_ns_cmd_read: buffer=%p", buffer);
    rc = spdk_nvme_ns_cmd_read(_device->ns(),
                               _qpair,
                               buffer,
                               lba, /* LBA start */
                               lba_count,        /* number of LBAs */
                               sync_io_complete, /* completion callback */
                               (void*)&status,   /* callback arg */
                               0 /* flags */);
  }
  else if (op == OP_FLAG_WRITE) {

#ifdef CONFIG_FLUSH_CACHES_ON_IO
    clflush_area(buffer, lba_count * _block_size);
#endif

    if (option_DEBUG)
      PLOG("spdk_nvme_ns_cmd_write: buffer=%p", buffer);
    rc = spdk_nvme_ns_cmd_write(_device->ns(),
                                _qpair,
                                buffer,
                                lba, /* LBA start */
                                lba_count,        /* number of LBAs */
                                sync_io_complete, /* completion callback */
                                (void*)&status,   /* callback arg */
                                0 /* flags */);
  }
  else {
    throw Logic_exception("invalid op");
  }

  if (rc != 0) {
    PERR("spdk_nvme_ns_cmd_xxx failed unexpectedly.");
    assert(0);
    return E_FAIL;
  }

  // poll for completion
  //
  while (status != TRANSMIT_STATUS_COMPLETE) {
    spdk_nvme_qpair_process_completions(_qpair, 0 /* unlimited completions */);
  }

  return S_OK;
}

/* --------- async operations ----------*/


static void
async_io_complete(void* arg, const struct spdk_nvme_cpl* completion)
{
  cb_arg * cbarg = static_cast<cb_arg*>(arg);
  cbarg->list->sp_enqueue(cbarg->tag);
  auto queue = cbarg->nvme_queue;
  assert(queue);
  
#ifdef CONFIG_QUEUE_STATS
  queue->_stats.total_io_cycles += (rdtsc() - cbarg->time_stamp);
#endif

  cbarg->used = false; /* free frame */
  queue->_status.outstanding--;
  assert(queue->_status.outstanding >= 0);
}

status_t
Nvme_queue::submit_async_op(void* buffer, uint64_t lba, uint64_t lba_count,
                            int op, uint64_t tag)
{
  assert(buffer);

  if(!check_aligned(buffer, CONFIG_IO_MEMORY_ALIGNMENT_REQUIREMENT)) {
    throw General_exception("%s : misgaligned buffer (%p)",__PRETTY_FUNCTION__, buffer);
  }

  if (option_DEBUG)
    PMAJOR("NVMe IO submit_async_op: lba=%ld lba_count=%ld, op=%d, tag=%lu",
           lba, lba_count, op, tag);

#ifdef CHECK_THREAD_VIOLATION
  if(_stats.pthread==0)
    _stats.pthread = pthread_self();
  
  if(!pthread_equal(_stats.pthread,pthread_self())) {
    PERR("thread violation! %p != %p",(void*)  _stats.pthread, (void*)pthread_self());
    _stats.pthread = pthread_self();
  }
#endif

#ifdef CONFIG_QUEUE_STATS
  if(lba_count > _stats.max_io_size_blocks)
    _stats.max_io_size_blocks = lba_count;
#endif
  
  // bounds check LBA
  if (lba + lba_count >= _max_lba)
    throw API_exception("lba out of bounds");

  // select call back arg from pre-allocated array (avoids heap allocator)
  cb_arg* arg = &_cbargs[_cbargs_index];
  if(arg->used)
    throw General_exception("cbarg frames exhausted!");
  
  _cbargs_index++;
  if(_cbargs_index == NUM_SUB_FRAMES) _cbargs_index = 0;
  arg->tag = tag;
  arg->list = &_completion_list;
  arg->used = true;
  arg->nvme_queue = this;
  
#ifdef CONFIG_QUEUE_STATS
  cpu_time_t start_time = arg->time_stamp = rdtsc();
#endif
  
  int rc = 0;
  if (op & OP_FLAG_READ) {
    rc = spdk_nvme_ns_cmd_read(_device->ns(),
                               _qpair,
                               buffer, /* va of payload */
                               lba, /* LBA start */
                               lba_count,         /* number of LBAs */
                               async_io_complete, /* completion callback */
                               (void*)arg,        /* callback arg */
                               0 /* flags */);
  }
  else if (op & OP_FLAG_WRITE) {

    wmb();
    
#ifdef CONFIG_FLUSH_CACHES_ON_IO
    clflush_area(buffer, lba_count * _block_size);
#endif

    rc = spdk_nvme_ns_cmd_write(_device->ns(),
                                _qpair,
                                buffer,
                                lba, /* LBA start */
                                lba_count,         /* number of LBAs */
                                async_io_complete, /* completion callback */
                                (void*)arg,        /* callback arg */
                                0 /* flags */);
  }
  else {
    throw Logic_exception("unexpected condition");
  }

  if(rc == -ENOMEM) {
    PERR("(direct mode) IO submission failed. No resources, queue full?");
    assert(0);
  }
  else {
    assert(_status.outstanding >= 0);
    _status.outstanding++;
  }
    
#ifdef CONFIG_QUEUE_STATS
  cpu_time_t duration = rdtsc() - start_time;
  if(duration > _stats.max_submit_cycles)
    _stats.max_submit_cycles = duration;

  _stats.total_submit_cycles+=duration;
  _stats.issued++;

  if(_stats.issued % CONFIG_STATS_REPORT_INTERVAL == 0) {
    
#define COL_OUTPUT
#ifdef COL_OUTPUT
    printf("%p,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%.2f,%.2f,%lu,%.2f\n",
           this, _status.outstanding, _stats.failed_polls, _stats.list_skips,
           _stats.issued, _stats.polls, _stats.max_submit_cycles, _stats.max_complete_cycles,
           (float)_stats.total_submit_cycles / (float) _stats.issued,
           (float)_stats.total_complete_cycles / (float) _stats.issued,
           _stats.max_io_size_blocks,
           (float)_stats.total_io_cycles / (float) _stats.issued);  
#else
           PLOG("COMANCHE: Queue (%p) outstanding(%lu), failed polls(%lu), skips(%lu),\
 issued(%lu), polls(%lu), maxsubmit(%lu), maxcompl(%lu), meansubmit(%.2f), \
meancompl(%.2f), maxiosizeblks(%lu), meancyclesperio(%.2f)",
                this, _status.outstanding, _stats.failed_polls, _stats.list_skips,
                _stats.issued, _stats.polls, _stats.max_submit_cycles, _stats.max_complete_cycles,
                (float)_stats.total_submit_cycles / (float) _stats.issued,
                (float)_stats.total_complete_cycles / (float) _stats.issued,
                _stats.max_io_size_blocks,
                (float)_stats.total_io_cycles / (float) _stats.issued
                );  
#endif
  }
#endif
  return rc;
}


/** 
 * Completion callback (internal) for queued mode
 * 
 * @param arg 
 * @param completion 
 */
static void
async_io_internal_complete(void* arg, const struct spdk_nvme_cpl* completion)
{
  queued_io_descriptor_t * desc = static_cast<queued_io_descriptor_t *>(arg);
  assert(desc->magic == 0x10101010);

#ifdef CONFIG_QUEUE_STATS
  desc->queue->_stats.total_io_cycles += (rdtsc() - desc->time_stamp);
#endif


  if(desc->cb)
    desc->cb(desc->tag, desc->arg); /* make call back if needed */

  delete desc;
}


void Nvme_queue::submit_async_op_internal(queued_io_descriptor_t * desc)
{
  assert(desc->buffer);
  assert(check_aligned(desc->buffer, CONFIG_IO_MEMORY_ALIGNMENT_REQUIREMENT));

  if (option_DEBUG)
    PLOG("submit_async_op_internal: queue=%p, lba=%ld lba_count=%ld, op=%d, tag=%d",
         this, desc->lba, desc->lba_count, desc->op, desc->tag);

#ifdef CONFIG_QUEUE_STATS
  if(desc->lba_count > _stats.max_io_size_blocks)
    _stats.max_io_size_blocks = desc->lba_count;
#endif

  // bounds check LBA
  if (desc->lba + desc->lba_count >= _max_lba)
    throw API_exception("lba out of bounds");

 retry_submission:

#ifdef CONFIG_QUEUE_STATS
  cpu_time_t start_time = desc->time_stamp = rdtsc();
  desc->queue = this;
#endif
  
  int rc = 0;
  if (desc->op == OP_FLAG_READ) {
    rc = spdk_nvme_ns_cmd_read(_device->ns(),
                               _qpair,
                               desc->buffer,
                               desc->lba, /* LBA start */
                               desc->lba_count,         /* number of LBAs */
                               async_io_internal_complete, /* completion callback */
                               (void*)desc,        /* callback arg */
                               0 /* flags */);
  }
  else if (desc->op == OP_FLAG_WRITE) {

    wmb();
    
#ifdef CONFIG_FLUSH_CACHES_ON_IO
    clflush_area(desc->buffer, desc->lba_count * _block_size);
#endif

    rc = spdk_nvme_ns_cmd_write(_device->ns(),
                                _qpair,
                                desc->buffer,
                                desc->lba, /* LBA start */
                                desc->lba_count,         /* number of LBAs */
                                async_io_internal_complete, /* completion callback */
                                (void*)desc,        /* callback arg */
                                0 /* flags */);
  }
  else {
    throw Logic_exception("unexpected condition");
  }

  if(rc == -ENOMEM) {
    // PERR("(queued mode) IO submission failed. No resources. ns=%p lba=%ld buffer=%p",
    //      _device->ns(), desc->lba, desc->buffer);
    process_completions();
    cpu_relax();
    goto retry_submission;
  }

  
#ifdef CONFIG_QUEUE_STATS
  cpu_time_t duration = rdtsc() - start_time;
  if(duration > _stats.max_submit_cycles)
    _stats.max_submit_cycles = duration;

  _stats.total_submit_cycles+=duration;
  _stats.issued++;

  if(_stats.issued % CONFIG_STATS_REPORT_INTERVAL == 0)
    PLOG("COMANCHE: Queue (%p) outstanding(%lu), failed polls(%lu), skips(%lu), issued(%lu), polls(%lu), maxsubmit(%lu), \
maxcompl(%lu), meansubmit(%.2f), meancompl(%.2f), maxiosizeblks(%lu), meancyclesperio(%.2f)",
         this, _status.outstanding, _stats.failed_polls, _stats.list_skips,
         _stats.issued, _stats.polls, _stats.max_submit_cycles, _stats.max_complete_cycles,
         (float)_stats.total_submit_cycles / (float) _stats.issued,
         (float)_stats.total_complete_cycles / (float) _stats.issued,
         _stats.max_io_size_blocks,
         (float)_stats.total_io_cycles / (float) _stats.issued
         );  
#endif
  
}

uint64_t
Nvme_queue::get_next_async_completion()
{
#ifdef CONFIG_QUEUE_STATS
  cpu_time_t start_time = rdtsc();
  _stats.polls++;
#endif

  int32_t rc = spdk_nvme_qpair_process_completions(_qpair,0); /* process all completions */
  if(rc < 0) throw General_exception("spdk_nvme_qpair_process_completions failed");

  if(_completion_list.empty()) {
    return 0;
  }

  uint64_t tag = 0;
  _completion_list.sc_dequeue(tag);

  //  PLOG("Nvme_queue; completing %d", tag);
  assert(tag > 0);
  if(tag != _status.last_tag+1)
    throw General_exception("unhandled out of order");

  _status.last_tag = tag;

  return tag;
}

uint64_t
Nvme_queue::get_last_completion()
{
#ifdef CONFIG_QUEUE_STATS
  cpu_time_t start_time = rdtsc();
  _stats.polls++;
#endif
  int32_t rc = spdk_nvme_qpair_process_completions(_qpair,0); /* process all completions */
  if(rc < 0) throw General_exception("spdk_nvme_qpair_process_completions failed");

  uint64_t tag = 0;
  while(_completion_list.sc_dequeue(tag)==0) {
    if(tag == _status.last_tag+1)
      _status.last_tag = tag;
    else
      throw General_exception("unhandled out of order");    
  }
  
  return _status.last_tag;
}
  

int32_t Nvme_queue::process_completions(int limit)
{
#ifdef CONFIG_QUEUE_STATS
  cpu_time_t start_time = rdtsc();
  _stats.polls++;
#endif

  int32_t nc = spdk_nvme_qpair_process_completions(_qpair,
                                                   limit /* max number of completions */);

  if(nc < 0) throw General_exception("spdk_nvme_qpair_process_completions failed");

#ifdef CONFIG_QUEUE_STATS
  if(nc == 0) {
    _stats.failed_polls++;
  }
  else {
    cpu_time_t duration = rdtsc() - start_time;
    if(duration > _stats.max_complete_cycles) {
      _stats.max_complete_cycles = duration;
    }
    _stats.total_complete_cycles += duration;
  }
#endif

  return nc;
}

size_t Nvme_queue::outstanding_count()
{
  process_completions(0);
  return _status.outstanding;
}


