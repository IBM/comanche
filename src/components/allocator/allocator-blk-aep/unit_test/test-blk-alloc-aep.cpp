/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include <mcheck.h>
#include <gtest/gtest.h>
#include <string>
#include <list>
#include <set>
#include <common/cycles.h>
#include <common/rand.h>
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
#include <api/block_allocator_itf.h>

#define PMEM_PATH "/mnt/pmem0/pool/0/"
#define DO_ERASE //erase blk allocation info?

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
  static size_t _nr_blks; //number of blocks from IBlockdev
};

Component::IBlock_device * Block_allocator_test::_block;
Component::IBlock_allocator * Block_allocator_test::_alloc;
size_t Block_allocator_test::_nr_blks;;

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

  _block = fact->create("86:00.0", &cpus);

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
  config_string += "\",\"size_in_blocks\":1000}";
  PLOG("config: %s", config_string.c_str());

  _nr_blks = 1000;

  _block = fact->create(config_string);
  assert(_block);
  fact->release_ref();
  PINF("Block-layer component loaded OK (itf=%p)", _block);

#endif
}

TEST_F(Block_allocator_test, InstantiateBlockAllocator)
{
  using namespace Component;
  
  IBase * comp = load_component("libcomanche-blkalloc-aep.so",
                                Component::block_allocator_aep_factory);
  assert(comp);
  IBlock_allocator_factory * fact = static_cast<IBlock_allocator_factory *>
    (comp->query_interface(IBlock_allocator_factory::iid()));

  size_t num_blocks = _nr_blks;
  PLOG("Opening allocator to support %lu blocks", num_blocks);
  _alloc = fact->open_allocator(
                                num_blocks,
                                PMEM_PATH,
                                "block-alloc-ut");  
  fact->release_ref();  
}
	

#if 1
TEST_F(Block_allocator_test, TestAllocation)
{
  struct Record {
    lba_t lba;
    void* handle;
  };

  std::vector<Record> v;
  std::set<lba_t> used_lbas;
  size_t n_blocks = _nr_blks;
  //Core::AVL_range_allocator ra(0, n_blocks*KB(4));
  
  PLOG("total blocks = %ld (%lx)", n_blocks, n_blocks); 
  for(unsigned long i=0;i<n_blocks;i++) {
    void * p;
    size_t s = 1; // (genrand64_int64() % 5) + 2;
    lba_t lba = _alloc->alloc(s,&p);    
    PLOG("lba=%lx", lba);
    
    ASSERT_TRUE(used_lbas.find(lba) == used_lbas.end()); // not already in set

    //ASSERT_TRUE(ra.alloc_at(lba, s) != nullptr); // allocate range
      
    used_lbas.insert(lba);
    //    PLOG("[%lu]: lba(%ld) allocated %ld blocks", i, lba, s);
    v.push_back({lba,p});

    if(i % 100 == 0) PLOG("allocations:%ld", i);
  }

  for(auto& e: v) {
    _alloc->free(e.lba, e.handle);
  }
      
}
#endif

#if 0
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

/* 
 * erase the allocation info
 * 
 */
#ifdef DO_ERASE
TEST_F(Block_allocator_test, EraseAllocator){
  _alloc->resize(0,0);
}
#endif

TEST_F(Block_allocator_test, ReleaseAllocator)
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
