/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include <common/logging.h>
#include <core/dpdk.h>
#include <core/physical_memory.h>
#include "cstdio"
#include "nupm/mcas_mod.h"

using namespace Component;
#define MB(x) (x<<20)

int main()
{
  status_t                rc;
  static constexpr size_t ALLOC_SIZE = MB(16);

  unsigned master_core = 0;
  bool is_primary = false;
  DPDK::eal_init(1024, master_core , is_primary);  // limit 1 GiB

  // allocate memory
  PINF("Press Any key to allocate memory in client");
  getchar();

  Core::Physical_memory mem_alloc;
  io_buffer_t           io_mem =
      mem_alloc.allocate_io_buffer(ALLOC_SIZE, 4096, NUMA_NODE_ANY);

  void *virt_addr = mem_alloc.virt_addr(io_mem);
  PINF("[master]: virt addr is %p, region size %lu", virt_addr, ALLOC_SIZE);

  // expose io memory
  nupm::Memory_token token = 1;
  rc                       = nupm::expose_memory(token, virt_addr, ALLOC_SIZE);
  if (rc != S_OK) throw General_exception("expose failed");


  PINF("region exposed, Press Any key to terminate client");
  getchar();
  rc = nupm::revoke_memory(token);
  if (rc != S_OK) throw General_exception("revoke expose failed");
  mem_alloc.free_io_buffer(io_mem);
  return 0;
}
