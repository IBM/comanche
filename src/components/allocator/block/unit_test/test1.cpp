#include <mcheck.h>
#include <gtest/gtest.h>
#include <string>
#include <list>
#include <set>
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/str_utils.h>
#include <common/logging.h>
#include <common/cpu.h>
#include <common/utils.h>
#include <common/dump_utils.h>
#include <core/avl_malloc.h>
#include <core/dpdk.h>
#include <component/base.h>

#include <api/components.h>
#include <api/block_itf.h>
#include <api/region_itf.h>
#include <api/pmem_itf.h>
#include <api/block_allocator_itf.h>

//#define USE_PAGED_PMEM // used paged pmem
#define USE_SPDK_NVME_DEVICE // use SPDK-NVME or POSIX-NVME

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Block_allocator_test : public ::testing::Test {

 protected:

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:
  
  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }
  
  // Objects declared here can be used by all tests in the test case
  static Component::IBlock_device *      _block;
  static Component::IBlock_allocator *   _alloc;
  static Component::IPager *             _pager;
  static Component::IPersistent_memory * _pmem;
};

Component::IBlock_device * Block_allocator_test::_block;
Component::IBlock_allocator * Block_allocator_test::_alloc;
Component::IPager * Block_allocator_test::_pager;
Component::IPersistent_memory * Block_allocator_test::_pmem;

TEST_F(Block_allocator_test, InitDPDK)
{
  DPDK::eal_init(512);
}

TEST_F(Block_allocator_test, InstantiateBlockDevice)
{
#ifdef USE_SPDK_NVME_DEVICE
  
  Component::IBase * comp = Component::load_component("libcomanche-blknvme.so",
                                                      Component::block_nvme_factory);

  assert(comp);
  PLOG("Block_device factory loaded OK.");
  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  
  cpu_mask_t cpus;
  cpus.add_core(2);

  _block = fact->create("8b:00.0", &cpus);

  assert(_block);
  fact->release_ref();
  PINF("Lower block-layer component loaded OK.");

#else
  
  Component::IBase * comp = Component::load_component("libcomanche-blkposix.so",
                                                      Component::block_posix_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  std::string config_string;
  config_string = "{\"path\":\"";
  //  config_string += "/dev/nvme0n1";1
  config_string += "./blockfile.dat";
  //  config_string += "\"}";
  config_string += "\",\"size_in_blocks\":10000}";
  PLOG("config: %s", config_string.c_str());

  _block = fact->create(config_string);
  assert(_block);
  fact->release_ref();
  PINF("Block-layer component loaded OK (itf=%p)", _block);

#endif
}

TEST_F(Block_allocator_test, InstantiatePmem)
{
  using namespace Component;

#ifdef USE_PAGED_PMEM
#define NUM_PAGER_PAGES 1
  {
    IBase * comp = load_component("libcomanche-pagersimple.so",
                                  Component::pager_simple_factory);
    assert(comp);
    IPager_factory * fact = static_cast<IPager_factory *>(comp->query_interface(IPager_factory::iid()));
    assert(fact);
    _pager = fact->create(NUM_PAGER_PAGES,"unit-test-heap",_block);
    assert(_pager);
    fact->release_ref();
  }
  {
  IBase * comp = load_component("libcomanche-pmempaged.so",
                                Component::pmem_paged_factory);
  assert(comp);
  IPersistent_memory_factory * fact = static_cast<IPersistent_memory_factory *>
    (comp->query_interface(IPersistent_memory_factory::iid()));
  assert(fact);
  _pmem = fact->open_allocator("testowner",_pager);
  assert(_pmem);
  fact->release_ref();                     
  }
  _pmem->start();

#else
  
  IBase * comp = load_component("libcomanche-pmemfixed.so",
                                Component::pmem_fixed_factory);
  assert(comp);
  IPersistent_memory_factory * fact = static_cast<IPersistent_memory_factory *>
    (comp->query_interface(IPersistent_memory_factory::iid()));
  assert(fact);
  _pmem = fact->open_allocator("unit-test-owner",_block);
  assert(_pmem);
  fact->release_ref();
#endif
  _pmem->start();
  
  PINF("Pmem-fixed component loaded OK.");
}


TEST_F(Block_allocator_test, InstantiateBlockAllocator)
{
  using namespace Component;
  
  IBase * comp = load_component("libcomanche-allocblock.so",
                                Component::block_allocator_factory);
  assert(comp);
  IBlock_allocator_factory * fact = static_cast<IBlock_allocator_factory *>
    (comp->query_interface(IBlock_allocator_factory::iid()));

  size_t num_blocks = GB(375)/KB(4);
  PLOG("Opening allocator to support %lu blocks", num_blocks);
  _alloc = fact->open_allocator(_pmem,
                                num_blocks,
                                "block-alloc-ut");  
  fact->release_ref();  
}

	

#if 0
TEST_F(Block_allocator_test, TestAllocation)
{
  struct Record {
    lba_t lba;
    void* handle;
  };

  std::vector<Record> v;
  std::set<lba_t> used_lbas;

  Core::AVL_range_allocator ra(0,GB(375)/KB(4));
  
  for(unsigned long i=0;i<1000;i++) {
    void * p;
    size_t s = (rand() % 2000) + 1;
    lba_t lba = _alloc->alloc(s,&p);    
    ASSERT_TRUE(used_lbas.find(lba) == used_lbas.end()); // not already in set

    ASSERT_TRUE(ra.alloc_at(lba, s) != nullptr); // allocate range
      
    used_lbas.insert(lba);
    PLOG("[%lu]: lba(%ld) allocated %ld blocks", i, lba, s);
    v.push_back({lba,p});
    i++;
  }

  for(auto& e: v) {
    _alloc->free(e.lba, e.handle);
  }
      
}
#endif

#if 1
TEST_F(Block_allocator_test, Exhaust)
{
  PLOG("each block 1024 bytes");
  unsigned long count = 0;
  while(1) {
    void * p;
    size_t s = 1;
    lba_t lba = _alloc->alloc(s,&p);
    if((++count % 10000) == 0) {
      PLOG("allocated: %lu allocations", count);
    }
  }
}
#endif

TEST_F(Block_allocator_test, ReleaseBlockDevice)
{
  ASSERT_TRUE(_alloc);
  ASSERT_TRUE(_block);

  _alloc->release_ref();
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  mtrace();
  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  muntrace();
  return r;
}
