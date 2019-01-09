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

#ifndef __NVME_BUFFER_H__
#define __NVME_BUFFER_H__

#include <assert.h>
#include <common/types.h>
#include <string.h>
#include <unistd.h>
#include "nvme_device.h"

enum {
  TRANSMIT_STATUS_COMPLETE = 0,
  TRANSMIT_STATUS_WRITE_INFLIGHT = 1,
  TRANSMIT_STATUS_READ_INFLIGHT = 1,
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
addr_t get_physical(void* io_buffer);

/**
 * Allocate an IO buffer
 *
 * @param size Size in bytes to allocate
 * @param numa_socket NUMA zone to allocate from (-1 is SOCKET_ANY)
 * @param zero_init Set to zero-out new memory
 *
 * @return Pointer to IO buffer
 */
void* allocate_io_buffer(size_t size, unsigned numa_socket = -1,
                         bool zero_init = false);

/**
 * Free IO buffer
 *
 * @param buffer Pointer to previously allocated IO buffer
 */
void free_io_buffer(void* buffer);

}  // namespace Nvme_buffer

#endif  // __NVME_BUFFER_H__
