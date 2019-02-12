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
