/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include <api/block_itf.h>
#include <api/components.h>
#include <common/logging.h>
#include <core/dpdk.h>
#include <core/physical_memory.h>
#include "cstdio"
#include "nupm/mcas_mod.h"

using namespace Component;
Component::IBlock_device *_block;
struct {
  std::string pci;
} opt;

status_t init_block_device()
{
  Component::IBase *comp = Component::load_component(
      "libcomanche-blknvme.so", Component::block_nvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory *fact = (IBlock_device_factory *) comp->query_interface(
      IBlock_device_factory::iid());
  cpu_mask_t cpus;
  cpus.add_core(2);
  //  cpus.add_core(25);

  _block = fact->create(opt.pci.c_str(), &cpus);

  assert(_block);
  fact->release_ref();
  PINF("nvme-based block-layer component loaded OK.");

  return S_OK;
}

status_t destroy_block_device()
{
  assert(_block);
  _block->release_ref();
  return S_OK;
}

int main()
{
  static constexpr size_t ALLOC_SIZE = MB(16);
  nupm::Memory_token      token      = 1;

  // map to mcas master memory
  size_t region_size = 0;
  void * virt_addr   = (void *) 0x700000000;
  void * rv          = nupm::mmap_exposed_memory(token, region_size, virt_addr);
  if (rv != virt_addr || region_size != ALLOC_SIZE)
    throw General_exception("mmap failed, got region_size = %lu, addr = %p",
                            region_size, rv);
  PINF("[master]: virt addr is %p, region size %lu", virt_addr, ALLOC_SIZE);

  // do block io
  init_block_device();

  PINF("Press Any key");
  getchar();
  destroy_block_device();
  return 0;
}
