#include <api/block_allocator_itf.h>
#include <api/kvstore_itf.h>
#include <common/utils.h>
#include <gtest/gtest.h>
#include <map>
#include "bitmap_ikv.h"
using namespace Component;
using namespace block_alloc_ikv;

namespace
{
// The fixture for testing class Foo.
class BitmapIkv_test : public ::testing::Test {
 protected:
  static IKVStore::pool_t _pool;
  static IKVStore *       _kvstore;
  static bitmap_ikv *     _bitmap;
  static std::string      _test_id;

  static std::map<unsigned int, unsigned int> _regions; /** pos->order*/
  static IKVStore::key_t                      _lockkey; /** for this 4Mbitmap*/
};

// static constexpr int _k_mid_order = 16;
static constexpr int _k_mid_order = 16;

IKVStore::pool_t BitmapIkv_test::_pool;
IKVStore *       BitmapIkv_test::_kvstore;
bitmap_ikv *     BitmapIkv_test::_bitmap;
std::string      BitmapIkv_test::_test_id = "thisistheiid";
IKVStore::key_t  BitmapIkv_test::_lockkey;

std::map<unsigned int, unsigned int> BitmapIkv_test::_regions;

TEST_F(BitmapIkv_test, Instantiate)
{
  /* create object instance through factory */
  Component::IBase *comp = Component::load_component(
      "libcomanche-storefile.so", Component::filestore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory *fact =
      (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  std::map<std::string, std::string> params;
  params["pm_path"]    = "/mnt/pmem0/";
  unsigned debug_level = 0;

  // this nvme-store use a block device and a block allocator
  _kvstore = fact->create(debug_level, params);

  fact->release_ref();

  PLOG(" test-nvmestore: try to openpool");
  ASSERT_TRUE(_kvstore);
  // pass blk and alloc here
  _pool = _kvstore->create_pool("block-alloc-pool", MB(128));
  ASSERT_TRUE(_pool > 0);

  _bitmap = new bitmap_ikv(_kvstore, _pool, _test_id);
  _bitmap->load(_lockkey);
  _bitmap->zero();
}

TEST_F(BitmapIkv_test, Alloc)
{
  // i should find 4096*1024/16 = 256
  size_t expected_regions = _bitmap->get_capacity() / (1 << _k_mid_order);
  PINF("Allcoate!, expect %lu regions", expected_regions);
  size_t nr_regions = 0;

  int pos;

  while (1) {
    pos = _bitmap->find_free_region(_k_mid_order);
    if (pos == -1) {
      PLOG("stop\n");
      break;
    }
    else {
      nr_regions += 1;
      _regions.emplace(pos, _k_mid_order);
      PDBG("allocate at pos %u", pos);
      PDBG("now %lu instances", nr_regions);
    }
    if (nr_regions > expected_regions) FAIL();
  }

  ASSERT_EQ(expected_regions, nr_regions);
}
TEST_F(BitmapIkv_test, Free)
{
  PINF("Free!");
  for (auto i = _regions.begin(); i != _regions.end(); i++) {
    unsigned int pos   = i->first;
    unsigned int order = i->second;

    _bitmap->release_region(pos, order);
  }
}

TEST_F(BitmapIkv_test, Finalize)
{
  _bitmap->flush(_lockkey);
  delete _bitmap;
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
