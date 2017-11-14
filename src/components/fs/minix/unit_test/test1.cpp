#include <gtest/gtest.h>
#include <string>
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/cpu.h>

#include <component/base.h>
#include <api/components.h>
#include <api/block_itf.h>
#include <api/fs_itf.h>

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Minix_fs_test : public ::testing::Test {

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
  static Component::IBlock_device * _block;
  static Component::IFile_system * _fs;
};


Component::IBlock_device * Minix_fs_test::_block;
Component::IFile_system * Minix_fs_test::_fs;

TEST_F(Minix_fs_test, InstantiateBlockDevice)
{
  Component::IBase * comp = Component::load_component("libcomanche-blknvme.so",
                                                      Component::block_nvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  cpu_mask_t cpus;
  cpus.add_core(2);
  cpus.add_core(3);

  _block = fact->create("01:00.0", &cpus);
  assert(_block);
  fact->release_ref();
  PINF("Lower block-layer component loaded OK.");
}

TEST_F(Minix_fs_test, InstantiateFs)
{
  assert(_block);

  Component::IBase * comp = Component::load_component("libcomanche-fsminix.so",
                                                      Component::fs_minix);
  assert(comp);
  _fs = (IFile_system *) comp->query_interface(IFile_system::iid());
  assert(_fs);
  
  PINF("Fs-minix component loaded OK.");

  int remaining = _fs->bind(_block);
  ASSERT_EQ(remaining, 0);

  PINF("Binding fs-minix to lower layer OK.");
}

TEST_F(Minix_fs_test, RunFuse)
{
  _fs->start();
}

TEST_F(Minix_fs_test, ReleaseBlockDevice)
{
  assert(_block);
  _fs->release_ref();
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
