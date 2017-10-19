/*
   Copyright [2017] [IBM Corporation]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
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
#include <rte_malloc.h>
#include <spdk/nvme.h>

#include "nvme_queue.h"
#include "nvme_device.h"

Nvme_queue::Nvme_queue(Nvme_device* device,
                       unsigned qid,
                       struct spdk_nvme_qpair* const qpair)
  : _device(device),
    _qpair(qpair)
{
  if(option_DEBUG)
    PLOG("New Nvme_queue: %u", qid);
  
  assert(device);
  assert(qpair);

  _ns_id = spdk_nvme_ns_get_id(device->ns());
  _queue_id = qid;
  _block_size = _device->get_block_size(1); // namespace id
  _max_lba = spdk_nvme_ns_get_num_sectors(device->ns());
  
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
}


/* --------- async operations ----------*/

/** 
 * Completion callback for IO operations
 * 
 * @param arg 
 * @param completion 
 */
static void
async_io_internal_complete(void* arg, const struct spdk_nvme_cpl* completion)
{
  IO_descriptor * desc = static_cast<IO_descriptor *>(arg);

#ifdef CONFIG_QUEUE_STATS
  desc->queue->_stats.total_io_cycles += (rdtsc() - desc->time_stamp);
#endif

  if(desc->cb)
    desc->cb(desc->tag, desc->arg0, desc->arg1); /* make call back if needed */

  desc->queue->remove_pending_fifo(desc);
  
  if(completion->status.sc != 0)
    PERR("IO error: tag=%lu", desc->tag);
  assert(desc);
  
  desc->queue->device()->free_desc(desc);
}


void Nvme_queue::submit_async_op_internal(IO_descriptor * desc)
{
  assert(desc);
  assert(desc->buffer);
  assert(check_aligned(desc->buffer, CONFIG_IO_MEMORY_ALIGNMENT_REQUIREMENT));

  if (option_DEBUG)
    PINF("[+] submit_async_op_internal: queue=%p, lba=%ld lba_count=%ld, op=%s, tag=%lu",
         this, desc->lba, desc->lba_count, desc->op==COMANCHE_OP_READ ? "R" : "W", desc->tag);

#ifdef CONFIG_QUEUE_STATS
  if(desc->lba_count > _stats.max_io_size_blocks)
    _stats.max_io_size_blocks = desc->lba_count;
#endif

  // bounds check LBA
  if (desc->lba + desc->lba_count >= _max_lba)
    throw API_exception("lba out of bounds (lba=%lu, max=%lu)",desc->lba + desc->lba_count, _max_lba);

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

// uint64_t
// Nvme_queue::get_next_async_completion()
// {
// #ifdef CONFIG_QUEUE_STATS
//   cpu_time_t start_time = rdtsc();
//   _stats.polls++;
// #endif

//   int32_t rc = spdk_nvme_qpair_process_completions(_qpair,0); /* process all completions */
//   if(rc < 0) throw General_exception("spdk_nvme_qpair_process_completions failed");

//   if(_completion_list.empty()) {
//     return 0;
//   }

//   _completion_list.sc_dequeue(tag);

//   //  PLOG("Nvme_queue; completing %d", tag);
//   assert(tag > 0);
//   if(tag != _status.last_tag+1) {
//     PERR("unhandled out of order");
//     exit(0);
//   }

//   _status.last_tag = tag;

//   return tag;
// }

// uint64_t
// Nvme_queue::get_last_completion()
// {
// #ifdef CONFIG_QUEUE_STATS
//   cpu_time_t start_time = rdtsc();
//   _stats.polls++;
// #endif
//   int32_t rc = spdk_nvme_qpair_process_completions(_qpair,0); /* process all completions */
//   if(rc < 0) throw General_exception("spdk_nvme_qpair_process_completions failed");

//   uint64_t tag = 0;
//   while(1) {

//     process_completions(0);
//     _completion_list.mc_dequeue(tag);
    
//     if(tag == _status.last_tag + 1) {
//       _status.last_tag = tag;
//       return _status.last_tag;
//     }
//     _completion_list.mp_enqueue(tag);
//     cpu_relax();
//   }
//   throw General_exception("get_last_completion spun");

// }
  

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

bool Nvme_queue::
check_completion(uint64_t gwid)
{
  uint64_t last = get_last_completion();
  //  PLOG("!!!! Check_completion last=%lu", last);
  if(last == UINT64_MAX) return true; // all complete!
  return (last >= gwid);
}
