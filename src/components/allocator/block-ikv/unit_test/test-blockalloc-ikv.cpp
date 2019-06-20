/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include <api/block_allocator_itf.h>
#include <api/kvstore_itf.h>
#include <common/utils.h>
#include <gtest/gtest.h>
#include <map>
#include <thread>
#include "bitmap_ikv.h"
using namespace Component;
using namespace block_alloc_ikv;

namespace
{
// The fixture for testing class Foo.
class BlkAllocIkv_test : public ::testing::Test {
 protected:
  static IKVStore::pool_t _pool;
  static IKVStore *       _kvstore;
  static std::string      _test_id;

  static Component::IBlock_allocator *_alloc;
  static size_t                       _nr_blocks;
};

IKVStore::pool_t BlkAllocIkv_test::_pool;
IKVStore *       BlkAllocIkv_test::_kvstore;
std::string      BlkAllocIkv_test::_test_id = "guesswhatid";

Component::IBlock_allocator *BlkAllocIkv_test::_alloc;
size_t                       BlkAllocIkv_test::_nr_blocks = 10000000;

TEST_F(BlkAllocIkv_test, Instantiate)
{
  // First Initialize ikvstore and one pool
  Component::IBase *comp = Component::load_component(
      "libcomanche-storefile.so", Component::filestore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory *fact =
      (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  std::map<std::string, std::string> params;
  params["pm_path"]    = "/mnt/pmem0/";
  unsigned debug_level = 0;

  _kvstore = fact->create(debug_level, params);
  fact->release_ref();

  PLOG(" test-nvmestore: try to openpool");
  ASSERT_TRUE(_kvstore);
  _pool = _kvstore->create_pool("block-alloc-pool", MB(128));
  ASSERT_TRUE(_pool > 0);

  // Then intialize the allocator
  comp = load_component("libcomanche-blkalloc-ikv.so",
                        Component::block_allocator_ikv_factory);
  assert(comp);
  IBlock_allocator_factory *fact_blk_alloc =
      static_cast<IBlock_allocator_factory *>(
          comp->query_interface(IBlock_allocator_factory::iid()));

  PLOG("Opening allocator to support %lu blocks", num_blocks);
  _alloc = fact_blk_alloc->open_allocator(_nr_blocks, _kvstore, _pool, _test_id,
                                          true);
  fact_blk_alloc->release_ref();
}

TEST_F(BlkAllocIkv_test, TestAllocation)
{
  size_t n_blocks = 1000;
  struct Record {
    lba_t lba;
    void *handle;
  };

  std::vector<Record> v;
  std::set<lba_t>     used_lbas;
  // Core::AVL_range_allocator ra(0, n_blocks*KB(4));

  PLOG("total blocks = %ld (%lx)", n_blocks, n_blocks);
  for (unsigned long i = 0; i < n_blocks; i++) {
    void * p;
    size_t s   = 1;  // (genrand64_int64() % 5) + 2;
    lba_t  lba = _alloc->alloc(s, &p);
    PLOG("lba=%lx", lba);

    ASSERT_TRUE(used_lbas.find(lba) == used_lbas.end());  // not already in set

    // ASSERT_TRUE(ra.alloc_at(lba, s) != nullptr); // allocate range

    used_lbas.insert(lba);
    //    PLOG("[%lu]: lba(%ld) allocated %ld blocks", i, lba, s);
    v.push_back({lba, p});

    if (i % 100 == 0) PLOG("allocations:%ld", i);
  }

  for (auto &e : v) {
    _alloc->free(e.lba, e.handle);
  }
}

static void allocate_bit(Component::IBlock_allocator *alloc, size_t num_bits)
{
  size_t                                n_blocks = num_bits;  // blocks
  std::chrono::system_clock::time_point _start, _end;

  struct Record {
    lba_t lba;
    void *handle;
  };

  std::vector<Record> v;

  _start = std::chrono::high_resolution_clock::now();

  for (unsigned long i = 0; i < n_blocks; i++) {
    void * p;
    size_t s   = 1;  // (genrand64_int64() % 5) + 2;
    lba_t  lba = alloc->alloc(s, &p);

    v.push_back({lba, p});
  }

  _end = std::chrono::high_resolution_clock::now();
  double secs =
      std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start)
          .count() /
      1000.0;
  PINF("*block allocator*: %lu allocations in %.1lf sec, rate: %2g", n_blocks,
       secs, ((double) n_blocks) / secs);

  /* TODO: free after join, otherwise freed lba will be reused, violate the test
   */
  for (auto &e : v) {
    alloc->free(e.lba, e.handle);
  }
}

TEST_F(BlkAllocIkv_test, TestConcurrentAlloc)
{
  int i;
  int nr_threads;

  size_t blocks_per_thread = 1000;
  PINF("how many work threads?");
  std::cin >> nr_threads;
  PINF("now starting %d working thread ...", nr_threads);

  std::vector<std::thread> threads;

  for (i = 0; i < nr_threads; i++) {
    threads.push_back(std::thread(allocate_bit, _alloc, blocks_per_thread));
  }

  for (auto &th : threads) th.join();
  PINF("*block allocator* all thread joined");
}

#if 0
TEST_F(BlkAllocIkv_test, EraseAllocator){
  _alloc->resize(0,0);
}
#endif

TEST_F(BlkAllocIkv_test, Finalize)
{
  ASSERT_TRUE(_alloc);

  _alloc->release_ref();

  _kvstore->close_pool(_pool);
  _kvstore->release_ref();
}

}  // namespace
int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
