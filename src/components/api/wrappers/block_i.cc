/*
   Copyright [2017-2019] [IBM Corporation]
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

#include <assert.h>
#include <common/exceptions.h>
#include <api/components.h>
#include "block_i.h"

IO_buffer::IO_buffer(Component::IBlock_device* owner, uint64_t iob) :
  _owner(owner), _iob(iob)
{
  TRACE();
  assert(_owner);
  assert(_iob);
}

IO_buffer::~IO_buffer() {
  assert(_owner);
  PINF("clearing up buffer");
  _owner->free_io_buffer(_iob);    
}

void IO_buffer::show() const {
  PINF("owner:%p; iob:0x%lx", _owner, _iob);
}

void IO_buffer::operator=(const IO_buffer& obj) {
  _owner = obj._owner;
  _iob = obj._iob;

}



Block::Block(const char * config, unsigned long cpu_mask, const char * lib_name)
{
  using namespace Component;

  IBase * comp;
  if(lib_name) 
    comp = lib_name ?
      load_component(lib_name, block_nvme_factory) :
      load_component("libcomanche-blknvme.so", block_nvme_factory);
 
  if(comp == nullptr)
    throw Comanche_exception("load_component failed");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  if(!fact) throw Comanche_exception("failed to get factory interface");

  cpu_mask_t mask;
  mask.set_mask(cpu_mask);

  try {
    _obj = fact->create(config,&mask);
  }
  catch(...) {
    throw Comanche_exception("create failed");
  }
  comp->release_ref();
}


Block::~Block()
{
  _obj->release_ref();
}

//   io_buffer_t IBlock__allocate_io_buffer(IBlock_ref ref, size_t size, unsigned alignment, int numa_node)
//   {
//     EXTRACT_OBJ;
//     return obj->allocate_io_buffer(size, alignment, numa_node);
//   }

//   status_t IBlock__realloc_io_buffer(IBlock_ref ref, io_buffer_t io_mem, size_t size, unsigned alignment)
//   {
//     EXTRACT_OBJ;
//     return obj->realloc_io_buffer(io_mem, size, alignment);
//   }

//   status_t IBlock__free_io_buffer(IBlock_ref ref, io_buffer_t io_mem)
//   {
//     EXTRACT_OBJ;
//     return obj->free_io_buffer(io_mem);
//   }

//   io_buffer_t IBlock__register_memory_for_io(IBlock_ref ref, void * vaddr, addr_t paddr, size_t len)
//   {
//     EXTRACT_OBJ;
//     return obj->register_memory_for_io(vaddr, paddr, len);
//   }

//   void IBlock__unregister_memory_for_io(IBlock_ref ref, void * vaddr, size_t len)
//   {
//     EXTRACT_OBJ;
//     obj->unregister_memory_for_io(vaddr, len);
//   }

//   void * IBlock__virt_addr(IBlock_ref ref, io_buffer_t buffer)
//   {
//     EXTRACT_OBJ;
//     return obj->virt_addr(buffer);
//   }

//   addr_t IBlock__phys_addr(IBlock_ref ref, io_buffer_t buffer)
//   {
//     EXTRACT_OBJ;
//     return obj->phys_addr(buffer);
//   }

//   size_t IBlock__get_size(IBlock_ref ref, io_buffer_t buffer)
//   {
//     EXTRACT_OBJ;
//     return obj->get_size(buffer);
//   }
  
//   workid_t IBlock__async_read(IBlock_ref ref,
//                               io_buffer_t buffer,
//                               uint64_t buffer_offset,
//                               uint64_t lba,
//                               uint64_t lba_count,
//                               int queue_id,
//                               io_callback_t cb,
//                               void * cb_arg0,
//                               void * cb_arg1)
//   {
//     EXTRACT_OBJ;
//     return obj->async_read(buffer, buffer_offset, lba, lba_count, queue_id, cb, cb_arg0, cb_arg1);
//   }

//   void IBlock__read(IBlock_ref ref,
//                     io_buffer_t buffer,
//                     uint64_t buffer_offset,
//                     uint64_t lba,
//                     uint64_t lba_count,
//                     int queue_id)
//   {
//     EXTRACT_OBJ;
//     obj->read(buffer, buffer_offset, lba, lba_count, queue_id);
//   }

//   workid_t IBlock__async_write(IBlock_ref ref,
//                                io_buffer_t buffer,
//                                uint64_t buffer_offset,
//                                uint64_t lba,
//                                uint64_t lba_count,
//                                int queue_id,
//                                io_callback_t cb,
//                                void * cb_arg0,
//                                void * cb_arg1)
//   {
//     EXTRACT_OBJ;
//     return obj->async_write(buffer, buffer_offset, lba, lba_count, queue_id, cb, cb_arg0, cb_arg1);
//   }

//   void IBlock__write(IBlock_ref ref,
//                      io_buffer_t buffer,
//                      uint64_t buffer_offset,
//                      uint64_t lba,
//                      uint64_t lba_count,
//                      int queue_id)
//   {
//     EXTRACT_OBJ;
//     obj->write(buffer, buffer_offset, lba, lba_count, queue_id);
//   }

//   bool IBlock__check_completion(IBlock_ref ref, workid_t gwid, int queue_id)
//   {
//     EXTRACT_OBJ;
//     return obj->check_completion(gwid, queue_id);
//   }

//   status_t IBlock__get_volume_info(IBlock_ref ref, VOLUME_INFO* devinfo)
//   {
//     EXTRACT_OBJ;
//     if(devinfo==NULL) return E_INVAL;
//     Component::VOLUME_INFO vi;
//     obj->get_volume_info(vi);
//     devinfo->block_size = vi.block_size;
//     devinfo->distributed = vi.distributed;
//     devinfo->hash_id = vi.hash_id;
//     devinfo->max_dma_len = vi.max_dma_len;
//     devinfo->max_lba = vi.max_lba;
//     devinfo->sw_queue_count = vi.sw_queue_count;
//     strncpy(devinfo->volume_name, vi.volume_name, VOLUME_INFO_MAX_NAME);
//     return S_OK;
//   }
// }
