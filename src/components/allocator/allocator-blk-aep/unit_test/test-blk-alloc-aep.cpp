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
#include <thread>
#include <chrono>
#include <gperftools/profiler.h>

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
#include "tbb/concurrent_unordered_set.h"

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
  static Component::IBlock_allocator *   _alloc;
  static size_t  _nr_blocks;
  //static tbb::concurrent_unordered_set<lba_t> _used_lbas;
};

Component::IBlock_allocator * Block_allocator_test::_alloc;
size_t Block_allocator_test::_nr_blocks = 1000000;
static tbb::concurrent_unordered_set<lba_t> _used_lbas;

#if 0
TEST_F(Block_allocator_test, InitDPDK)
{
  DPDK::eal_init(512);
}
#endif

TEST_F(Block_allocator_test, InstantiateBlockAllocator)
{
  size_t num_blocks = _nr_blocks;
  using namespace Component;
  
  IBase * comp = load_component("libcomanche-blkalloc-aep.so",
                                Component::block_allocator_aep_factory);
  assert(comp);
  IBlock_allocator_factory * fact = static_cast<IBlock_allocator_factory *>
    (comp->query_interface(IBlock_allocator_factory::iid()));

  PLOG("Opening allocator to support %lu blocks", num_blocks);
  _alloc = fact->open_allocator(
                                num_blocks,
                                PMEM_PATH,
                                "block-alloc-ut");  
  fact->release_ref();  
}
	

#if 0
TEST_F(Block_allocator_test, TestAllocation)
{
  size_t n_blocks = 100000;
  struct Record {
    lba_t lba;
    void* handle;
  };

  std::vector<Record> v;
  std::set<lba_t> used_lbas;
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

static void allocate_bit(Component::IBlock_allocator * alloc, size_t num_bits){
  size_t n_blocks = num_bits;// blocks
  std::chrono::system_clock::time_point _start, _end;

  struct Record {
    lba_t lba;
    void* handle;
  };

  std::vector<Record> v;
  //Core::AVL_range_allocator ra(0, n_blocks*KB(4));
  
  //PLOG("total blocks = %ld (%lx)", n_blocks, n_blocks); 
  ProfilerRegisterThread();

  _start = std::chrono::high_resolution_clock::now();

  for(unsigned long i=0;i<n_blocks;i++) {
    void * p;
    size_t s = 1; // (genrand64_int64() % 5) + 2;
    lba_t lba = alloc->alloc(s,&p);    
    //PLOG("lba=%lx", lba);
    
    //ASSERT_TRUE(used_lbas.find(lba) == used_lbas.end()); // not already in set
    //used_lbas.insert(lba);
    auto ret_insertion = _used_lbas.insert(lba);
    ASSERT_TRUE(ret_insertion.second);

    //    PLOG("[%lu]: lba(%ld) allocated %ld blocks", i, lba, s);
    v.push_back({lba,p});

    //if(i % 1000 == 0) PLOG("allocations:%ld", i);
  }

  _end = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / 1000.0;
  PINF("*block allocator*: %lu allocations in %.1lf sec, rate: %2g",n_blocks, secs,  ((double) n_blocks) / secs);

  for(auto& e: v) {
    alloc->free(e.lba, e.handle);
  }
}

TEST_F(Block_allocator_test, TestConcurrentAlloc)
{
  int i;
  int nr_threads;

  PINF("how many work threads?");
  std::cin >> nr_threads;
  PINF("now starting %d working thread ...", nr_threads);
  
  size_t blocks_per_thread = _nr_blocks/nr_threads; //_nr_blocks/nr_threads;
  std::vector<std::thread> threads;

  char profile_name[60];
  sprintf(profile_name, "bitmap-alloc-thread-%d.profile", nr_threads);
  ProfilerStart(profile_name);
  for(i = 0; i< nr_threads; i++){
    threads.push_back(std::thread(allocate_bit, _alloc,  blocks_per_thread));
  }

  for (auto& th : threads) th.join();
  PINF("*block allocator* all thread joined");

  ProfilerStop();
}

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

  _alloc->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  mtrace();
  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  muntrace();
  return r;
}
