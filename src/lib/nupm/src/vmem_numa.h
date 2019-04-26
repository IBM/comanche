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
#ifndef __NUPM_NUMA_VMEM__
#define __NUPM_NUMA_VMEM__

#include <common/exceptions.h>
#include <common/utils.h>
#include <libvmem.h>
#include "nd_utils.h"

namespace nupm
{
/**
 * Allocator based on libvmem (i.e. using AEP as volatile memory) but that
 * supports
 * NUMA-aware allocations.  Public methods on this class are thread safe.
 *
 */
class Vmem_allocator : private ND_control {
 private:
  static constexpr unsigned MAX_NUMA_SOCKETS = 2;

 public:
  /**
   * Constructor
   *
   */
  Vmem_allocator();

  /**
   * Destructor
   *
   */
  virtual ~Vmem_allocator();

  /**
   * Allocate a region of memory from specific numa socket
   *
   * @param numa_node NUMA socket counting from 0.  -1 = any
   * @param size Size of allocation in bytes
   *
   * @return Pointer to allocation
   */
  void *alloc(int numa_node, size_t size);

  /**
   * Free a previously allocated memory region
   *
   * @param ptr
   */
  void free(void *ptr);

 private:
  VMEM *vmem_for(void *ptr);

 private:
  VMEM *        _vmm[MAX_NUMA_SOCKETS];
  unsigned long _vmm_bases[MAX_NUMA_SOCKETS];
  unsigned long _vmm_ends[MAX_NUMA_SOCKETS];
};
}  // namespace nupm

#endif
