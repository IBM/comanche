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
using namespace Component;
namespace
{
// The fixture for testing class Foo.
class BlkAllocIkv_test : public ::testing::Test {
 protected:
  static IKVStore::pool_t _pool;
  static IKVStore *       _kvstore;
};

IKVStore::pool_t BlkAllocIkv_test::_pool;
IKVStore *       BlkAllocIkv_test::_kvstore;

TEST_F(BlkAllocIkv_test, Instantiate)
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
}

TEST_F(BlkAllocIkv_test, OpenPool)
{
  PLOG(" test-nvmestore: try to openpool");
  ASSERT_TRUE(_kvstore);
  // pass blk and alloc here
  _pool = _kvstore->create_pool("block-alloc-pool", MB(128));
  ASSERT_TRUE(_pool > 0);
}

TEST_F(BlkAllocIkv_test, Finalize)
{
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
