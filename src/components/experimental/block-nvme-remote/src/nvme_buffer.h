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

#ifndef __NVME_BUFFER_H__
#define __NVME_BUFFER_H__

#include <assert.h>
#include <common/types.h>
#include <unistd.h>
#include <string.h>
#include "nvme_device.h"

enum {
  TRANSMIT_STATUS_COMPLETE       = 0,
  TRANSMIT_STATUS_WRITE_INFLIGHT = 1,
  TRANSMIT_STATUS_READ_INFLIGHT  = 1,
};

namespace Nvme_buffer
{
/** 
 * Get physical address of IO buffer
 * 
 * @param io_buffer IO buffer
 * 
 * @return Physical address
 */
addr_t get_physical(void * io_buffer);

/** 
 * Allocate an IO buffer
 * 
 * @param size Size in bytes to allocate
 * @param numa_socket NUMA zone to allocate from (-1 is SOCKET_ANY)
 * @param zero_init Set to zero-out new memory
 * 
 * @return Pointer to IO buffer
 */
void * allocate_io_buffer(size_t size, unsigned numa_socket = -1, bool zero_init = false);

/** 
 * Free IO buffer
 * 
 * @param buffer Pointer to previously allocated IO buffer
 */
void free_io_buffer(void * buffer);

}


#endif  // __NVME_BUFFER_H__
