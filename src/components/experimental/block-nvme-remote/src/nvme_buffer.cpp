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

#include "nvme_buffer.h"

#include <spdk/nvme.h>
extern "C"
{
#include <rte_config.h>
#include <rte_malloc.h>
#include <rte_mempool.h>
}


addr_t Nvme_buffer::get_physical(void * io_buffer)
{
  return rte_malloc_virt2phy(io_buffer);
}


void * Nvme_buffer::allocate_io_buffer(size_t size_to_allocate,
                                       unsigned numa_socket,
                                       bool zero_init)
{
  assert(size_to_allocate % 64 == 0);

  void * ptr;
  
  if (zero_init) {
    ptr = rte_zmalloc_socket(NULL,
                             size_to_allocate,
                             4096 /*alignment*/,
                             numa_socket);
  }
  else {
    ptr = rte_malloc_socket(NULL,
                            size_to_allocate,
                            4096 /*alignment*/,
                            numa_socket);
  }

  //  PLOG("allocated Nvme_buffer @ phys:%lx", rte_malloc_virt2phy(ptr));

  if (!ptr)
    throw new Constructor_exception("rte_zmalloc failed in Nvme_buffer::allocate_io_buffer");

  return ptr;
}
  
void Nvme_buffer::free_io_buffer(void * buffer)
{
  assert(buffer);
  rte_free(buffer);
}

