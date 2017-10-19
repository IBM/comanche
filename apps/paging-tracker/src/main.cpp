#include <memory>

#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/str_utils.h>
#include <common/logging.h>
#include <common/cpu.h>
#include <common/utils.h>
#include <component/base.h>

#include <core/slab.h>
#include <core/avl_malloc.h>

#include <api/components.h>
#include <api/block_itf.h>
#include <api/region_itf.h>
#include <api/pager_itf.h>
#include <api/pmem_itf.h>

using namespace std;
using namespace Component;

// forward decls
IBlock_device * create_block_device();
IPager * create_pager(IBlock_device * block_device);
IPersistent_memory * create_pmem(IPager * pager);

static constexpr size_t METADATA_SIZE = MB(128);
int main(int argc, char * argv[])
{
  /* instantiate components */
  Itf_ref<IBlock_device> block_device(create_block_device());
  Itf_ref<IPager> pager(create_pager(block_device.get()));
  Itf_ref<IPersistent_memory> pmem(create_pmem(pager.get()));
  pmem->start();
  
  void * p = nullptr;
  bool reused;
  auto pma = pmem->open("myTree-md", METADATA_SIZE, NUMA_NODE_ANY, reused, p);
  PLOG("Pmem area was %s reused", reused ? "indeed" : "NOT");

  if(argc > 1) // option on command line will force reinit
    reused = false;
  
  /* slab allocator manages the nodes of the tree */
  Core::Slab::Allocator<Core::Memory_region>
    slab(p, METADATA_SIZE, "myTree-slab", !reused /* as_new */);

  PLOG("Slab: used=%ld", slab.used_slots());

  Core::AVL_range_allocator tree(slab,
                                 0x0, // start
                                 GB(375));

  unsigned long total = 0;
  while(total < 100000) {
    tree.alloc(4096, 4096);

    if(total++ % 1000 == 0)
      PLOG("%ld pages allocated", total);
  }

  PLOG("Slab: used=%ld", slab.used_slots());
    
  pmem->persist(pma);
  return 0;
}



IPersistent_memory * create_pmem(IPager * pager)
{
  IBase * comp = load_component("libcomanche-pmempaged.so",
                                Component::pmem_paged_factory);
  assert(comp);
  IPersistent_memory_factory * fact =
    static_cast<IPersistent_memory_factory *>
    (comp->query_interface(IPersistent_memory_factory::iid()));
  
  assert(fact);
  IPersistent_memory * pmem = fact->open_allocator("appowner", pager);
  assert(pmem);
  fact->release_ref();

  pmem->start();
  return pmem;
}

IBlock_device * create_block_device()
{
  IBase * comp = Component::load_component("libcomanche-blknvme.so",
                                           block_nvme_factory);

  assert(comp);
  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());

  cpu_mask_t mask;
  mask.add_core(2);
  auto component = fact->create("8b:00.0", &mask);

  assert(component);
  fact->release_ref();
  return component;
}

#define NUM_PAGER_PAGES 8

IPager * create_pager(IBlock_device * block_device)
{
  assert(block_device);
  
  IBase * comp = load_component("libcomanche-pagersimple.so",
                                Component::pager_simple_factory);
  assert(comp);
  IPager_factory * fact = static_cast<IPager_factory *>(comp->query_interface(IPager_factory::iid()));
  assert(fact);
  IPager * pager = fact->create(NUM_PAGER_PAGES,
                                "paging-tracker-heap",
                                block_device);
  assert(pager);
  return pager;
}

