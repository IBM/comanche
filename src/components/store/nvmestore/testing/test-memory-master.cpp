/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include <api/components.h>
#include <common/logging.h>
#include "cstdio"
#include "nupm/mcas_mod.h"

#include <api/block_itf.h>
#include <core/dpdk.h>
#include <core/physical_memory.h>

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

int main(int argc, char ** argv)
{
  if (argc != 2) {
    PINF("test <pci-address>");
    return 0;
  }

  opt.pci = argv[1];


  unsigned master_core = 0;
  bool is_primary = true;
  DPDK::eal_init(1024, master_core, is_primary);  // limit 1 GiB
  static constexpr size_t ALLOC_SIZE = MB(16);
  nupm::Memory_token      token      = 1;

  // do block io
  init_block_device();

  PINF("please start client and  Press Any key to map to it");
  getchar();

  // map to mcas master memory
  size_t region_size = 0;
  void * virt_addr   = (void *) 0x700000000;
  void * rv          = nupm::mmap_exposed_memory(token, region_size, virt_addr);
  if (rv != virt_addr || region_size != ALLOC_SIZE)
    throw General_exception("mmap failed, got region_size = %lu, addr = %p",
                            region_size, rv);
  PINF("[master master ok]: virt addr is %p, region size %lu", virt_addr, ALLOC_SIZE);

  PINF("Press Any key to exit master");
  getchar();

  destroy_block_device();
  return 0;
}
