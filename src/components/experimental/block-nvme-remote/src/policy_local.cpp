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
#include <city.h>
#include <spdk/env.h>
#include <common/assert.h>
#include <rte_malloc.h>
#include "policy_local.h"
#include "storage_device.h"
#include "nvme_device.h"
#include "nvme_queue.h"
#include "volume_agent_session.h"

Policy_local::Policy_local(Storage_device * storage_device) :
  _device(storage_device)
{
  assert(storage_device);
    
  _local_volume = storage_device->allocate_queue();
  assert(_local_volume);
}


Policy_local::~Policy_local()
{
  PLOG("Policy_local: dtor");
  /* don't delete storage device */
  delete _local_volume;
}

bool Policy_local::check_completion(uint64_t gwid)
{
  return (_local_volume->get_last_completion() >= gwid);
}

io_memory_t Policy_local::allocate_io_buffer(size_t size, size_t alignment, int numa_node)
{
  /* allocate RTE (DPDK) memory */
  return reinterpret_cast<io_memory_t>(rte_malloc_socket("Policy_local", size, alignment, numa_node));
}

status_t Policy_local::reallocate_io_buffer(io_memory_t io_mem, size_t size, unsigned alignment)
{
  assert(io_mem);
  void * ptr = reinterpret_cast<void*>(io_mem);
  void * newptr = rte_realloc(ptr,size,alignment);
  if(newptr==nullptr)
    return E_NO_MEM;

  if(newptr!=ptr) throw Logic_exception("pointer mismatch after reallocate_io_buffer");
  return reinterpret_cast<io_memory_t>(newptr);
}

io_memory_t Policy_local::register_io_buffer(void * ptr, size_t size)
{
  spdk_mem_register(ptr,size);
  return reinterpret_cast<io_memory_t>(ptr);
}

status_t Policy_local::unregister_io_buffer(void * ptr, size_t size) {
  spdk_mem_unregister(ptr, size);
  return S_OK;
}

void * Policy_local::get_addr(io_memory_t io_mem)
{
  return reinterpret_cast<void*>(io_mem);
}

uint64_t Policy_local::get_phys_addr(io_memory_t io_mem)
{
  return spdk_vtophys(reinterpret_cast<void*>(io_mem));
}

status_t Policy_local::free_io_buffer(io_memory_t io_mem)
{
  assert(io_mem > 0); 
  rte_free(reinterpret_cast<void*>(io_mem));
  return S_OK;
}

uint64_t Policy_local::issue_op(struct IO_command* cmd)
{
  /* bounds check */
  if((cmd->lba + cmd->lba_count + cmd->offset) > _local_volume->max_lba())
    throw API_exception("Policy_local: issue out of bounds (lba=%lu, lba_count=%lu, offset=%lu)",
                        cmd->lba, cmd->lba_count, cmd->offset);

  byte * payload = reinterpret_cast<byte*>(cmd->mrdesc);
  assert(payload);
  payload += cmd->offset * _local_volume->block_size();
  
  static uint64_t gwid = 0;
  gwid++;
  
  status_t rc = _local_volume->submit_async_op(payload,
                                               cmd->lba,
                                               cmd->lba_count,
                                               cmd->op_flags,
                                               gwid);
  if(rc!=S_OK)
    throw General_exception("local volume async op call failed in Policy_local::issue_op");

  return gwid;
}


uint64_t Policy_local::gratuitous_completion()
{
  _local_volume->process_completions(0);
  return _local_volume->get_last_completion();
}

void Policy_local::get_volume_info(Component::VOLUME_INFO& devinfo)
{
  devinfo.block_size = _local_volume->block_size();
  devinfo.max_lba = _local_volume->max_lba();
  devinfo.distributed = 0;
  devinfo.hash_id = _device->nvme_device()->get_serial_hash();
}


