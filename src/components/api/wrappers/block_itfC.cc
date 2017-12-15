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

#include <assert.h>
#include <common/exceptions.h>
#include <api/components.h>
#include "../block_itf.h"
#include "block_itfC.h"

extern "C"
IBlock_ref IBlock_factory__create(const char * config,
                                  unsigned long cpu_mask,
                                  const char * lib_name)
{
  using namespace Component;

  IBase * comp;
  if(lib_name) 
    comp = lib_name ?
      load_component(lib_name, block_nvme_factory) :
      load_component("libcomanche-blknvme.so", block_nvme_factory);
 
  assert(comp);

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  if(!fact) throw API_exception("failed to get factory interface");

  cpu_mask_t mask;
  auto obj = fact->create(config,&mask);
  comp->release_ref();
  return {obj};
}

#define EXTRACT_OBJ assert(ref.obj); \
  auto obj = static_cast<Component::IBlock_device*>(ref.obj);


extern "C"
status_t IBlock__release(IBlock_ref ref)
{
  if(!ref.obj) return E_INVAL;
  static_cast<Component::IBlock_device*>(ref.obj)->release_ref();
  return S_OK;
}

extern "C"
io_buffer_t IBlock__allocate_io_buffer(IBlock_ref ref, size_t size, unsigned alignment, int numa_node)
{
  EXTRACT_OBJ;
  return obj->allocate_io_buffer(size, alignment, numa_node);
}

extern "C"
status_t IBlock__realloc_io_buffer(IBlock_ref ref, io_buffer_t io_mem, size_t size, unsigned alignment)
{
  EXTRACT_OBJ;
  return obj->realloc_io_buffer(io_mem, size, alignment);
}

extern "C"
status_t IBlock__free_io_buffer(IBlock_ref ref, io_buffer_t io_mem)
{
  EXTRACT_OBJ;
  return obj->free_io_buffer(io_mem);
}

extern "C"
io_buffer_t IBlock__register_memory_for_io(IBlock_ref ref, void * vaddr, addr_t paddr, size_t len)
{
  EXTRACT_OBJ;
  return obj->register_memory_for_io(vaddr, paddr, len);
}

extern "C"
void IBlock__unregister_memory_for_io(IBlock_ref ref, void * vaddr, size_t len)
{
  EXTRACT_OBJ;
  obj->unregister_memory_for_io(vaddr, len);
}

extern "C"
void * IBlock__virt_addr(IBlock_ref ref, io_buffer_t buffer)
{
  EXTRACT_OBJ;
  return obj->virt_addr(buffer);
}

extern "C"
addr_t IBlock__phys_addr(IBlock_ref ref, io_buffer_t buffer)
{
  EXTRACT_OBJ;
  return obj->phys_addr(buffer);
}

extern "C"
size_t IBlock__get_size(IBlock_ref ref, io_buffer_t buffer)
{
  EXTRACT_OBJ;
  return obj->get_size(buffer);
}
