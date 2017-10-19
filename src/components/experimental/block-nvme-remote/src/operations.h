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

#ifndef __OPERATIONS_H__
#define __OPERATIONS_H__

#include "protocols.h"
#include "bitset.h"
#include "dd-config.h"
#include "types.h"

//#define NULL_IO // nullify NVMe IO for testing


/** 
 * Data operations should provide this interface.
 * 
 */
class Data_operation
{
public:
  /** 
   * Submit a memory region for asynchronous processing.
   * 
   * @param mr Memory region
   * @param wid Work identifier
   */
  virtual void submit(channel_memory_t mr)=0;

  /** 
   * Check for work item completion
   * 
   * 
   * @return 0 if none, work id otherwise.
   */
  virtual channel_memory_t get_next_completion()=0;
};



/** 
 * Null_IO_operation used for performance testing to
 * remove IO device processing
 * 
 */
class Null_IO_operation : public Data_operation
{
private:
  static constexpr bool option_DEBUG = false;

public:
  Null_IO_operation() :_wid_ring("null-io-rb",256) {
  }
  
  void submit(channel_memory_t mr) {
    _wid_ring.sp_enqueue(mr);
  }

  channel_memory_t get_next_completion() {
    channel_memory_t mr = nullptr;
    _wid_ring.sc_dequeue(mr);
    return mr;
  }

private:
  DPDK::Ring_buffer<channel_memory_t> _wid_ring;

};


/** 
 * Nvme_IO_operation class is used to perform async IO
 * operations
 * 
 */
class Nvme_IO_operation : public Data_operation
{
private:
  static constexpr bool option_DEBUG = false;

public:
  Nvme_IO_operation(Nvme_queue * queue) : _queue(queue) {
    assert(queue);
  }

  /** 
   * Submit an operation and memory payload to NVMe IO operation
   * 
   * @param mr Memory region (IO_command + payload)
   */
  void submit(channel_memory_t mr) {
    struct IO_command * cmd = static_cast<struct IO_command *>(mr->addr);
    assert(cmd);
    
    if(option_DEBUG) {
      PLOG("Nvme_IO_operation channel cmd: magic:%X op_flags:%X lba:%lu count:%lu gwid:%lu",
           cmd->magic, cmd->op_flags, cmd->lba, cmd->lba_count, cmd->gwid);
    }

    if((cmd->lba_count * IO_BLOCK_SIZE) > mr->length)
      throw Logic_exception("lba_count (%ld) too big for MR size (%ld) on channel",
                            cmd->lba_count,
                            mr->length);

    char* payload = (char*) (((char *) cmd) + sizeof(struct IO_command));
    payload += cmd->offset;
    
    /* async NVMe IO operation */    
    status_t rc = _queue->submit_async_op(payload,
                                          cmd->lba,
                                          cmd->lba_count,
                                          cmd->op_flags,
                                          ((uint64_t) mr));
    if(rc!=S_OK)
      throw General_exception("IO_operation device op: status=%d", rc);
  }

  /** 
   * Get next completed memory region
   * 
   * 
   * @return Completed memory region
   */
  channel_memory_t get_next_completion() {
    return reinterpret_cast<channel_memory_t>(_queue->get_next_async_completion());
  }

private:
  Nvme_queue * _queue;
  uint64_t _last_wid = 0;

};



#endif // __OPERATIONS_H__
