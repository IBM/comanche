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
#include <spdk/env.h>
#include <common/assert.h>
#include "policy_mirror.h"
#include "storage_device.h"
#include "nvme_device.h"
#include "nvme_queue.h"
#include "volume_agent_session.h"

Policy_mirror::Policy_mirror(Storage_device * storage_device,
                             Volume_agent_session * remote_volume) :
  _remote_volume(remote_volume),
  _device(storage_device)
{
  assert(storage_device);
  assert(remote_volume);
    
  _local_volume = storage_device->allocate_queue();
  assert(_local_volume);
}


Policy_mirror::~Policy_mirror()
{
  PLOG("Policy_mirror: dtor");
  /* don't delete storage device */
  delete _local_volume;
  delete _remote_volume;
}

bool Policy_mirror::check_completion(uint64_t gwid)
{
  return (_remote_volume->check_completion(gwid) &&
          _local_volume->get_last_completion() >= gwid);
}

io_memory_t Policy_mirror::allocate_io_buffer(size_t size, size_t alignment, int /*numa_node*/)
{
  /* allocate RTE (DPDK) memory and register with channel's RDMA engine */
  return reinterpret_cast<io_memory_t>(_remote_volume->alloc_region(size,alignment));
}

status_t Policy_mirror::reallocate_io_buffer(io_memory_t io_mem, size_t size, unsigned alignment)
{
  return E_NOT_IMPL;
}

io_memory_t Policy_mirror::register_io_buffer(void * ptr, size_t size) {
  spdk_mem_register(ptr, size);
  return reinterpret_cast<io_memory_t>(_remote_volume->register_region(ptr, size));
}

status_t Policy_mirror::unregister_io_buffer(void * ptr, size_t size) {
  spdk_mem_unregister(ptr, size);
  PWRN("Policy_mirror::unregister_io_buffer does not unregister with RDMA transport");
  return S_OK;
}

status_t Policy_mirror::free_io_buffer(io_memory_t io_mem)
{
  assert(io_mem > 0); 
  _remote_volume->free_region(reinterpret_cast<channel_memory_t>(io_mem));
  return S_OK;
}

uint64_t Policy_mirror::get_phys_addr(io_memory_t io_mem) 
{
  auto mr = reinterpret_cast<channel_memory_t>(io_mem);
  return spdk_vtophys(mr->addr);
}


uint64_t Policy_mirror::issue_op(struct IO_command* cmd)
{
  channel_memory_t mr = reinterpret_cast<channel_memory_t>(cmd->mrdesc);

#ifdef CONFIG_CHECKS_VALIDATE_POINTERS
  if(!check_ptr_valid(mr,sizeof(mr))) throw Program_exception("memory corruption");
#endif

  if(cmd->offset > 0)
    throw API_exception("Mirror policy does not support buffer offset parameter");

  /* bounds check */
  if((cmd->lba + cmd->lba_count + cmd->offset) > _local_volume->max_lba())
    throw API_exception("local issue out of bounds");
  
  /* mirror IO OP to local and remote */
  uint64_t issued_gwid = _remote_volume->submit_async(mr,
                                                      cmd->lba,
                                                      cmd->lba_count,
                                                      cmd->op_flags);
    
  status_t rc = _local_volume->submit_async_op(mr->addr,
                                               cmd->lba,
                                               cmd->lba_count,
                                               cmd->op_flags,
                                               issued_gwid /* pair up the tag */);
  if(rc!=S_OK)
    throw General_exception("local volume async op call failed in Client_session::async_submit");

  return issued_gwid;
}


uint64_t Policy_mirror::gratuitous_completion()
{
  _remote_volume->poll_outstanding();
  _local_volume->process_completions(0);
  uint64_t last_l_gwid = _local_volume->get_last_completion();
  uint64_t last_r_gwid = _remote_volume->last_completion();
  //      PLOG("last completions: %lu %lu",last_l_gwid, last_r_gwid);
  return std::min(last_l_gwid, last_r_gwid);    
}

void Policy_mirror::get_volume_info(Component::VOLUME_INFO& devinfo)
{
  /* for the moment use the local volume */
  devinfo.block_size = _local_volume->block_size();
  devinfo.max_lba = _local_volume->max_lba();
  devinfo.distributed = 1;
  /* for the moment just use local device sn : TODO combine with remote */
  devinfo.hash_id = _device->nvme_device()->get_serial_hash();
}

