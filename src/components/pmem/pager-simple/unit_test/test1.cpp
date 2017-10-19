#include <gtest/gtest.h>
#include <string>
#include <list>
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/str_utils.h>
#include <common/logging.h>
#include <common/cpu.h>

#include <component/base.h>
#include <api/components.h>
#include <api/block_itf.h>
#include <api/pager_itf.h>

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Simple_pager_test : public ::testing::Test {

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
  static Component::IPager *        _pager;
};


Component::IBlock_device * Simple_pager_test::_block;
Component::IPager * Simple_pager_test::_pager;

TEST_F(Simple_pager_test, InstantiateBlockDevice)
{
  Component::IBase * comp = Component::load_component("libcomanche-blkposix.so",
                                                      Component::block_posix_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());  
  _block = fact->create("{\"path\":\"blockdev.dat\", \"size_in_blocks\":4000 }");
  assert(_block);
  fact->release_ref();
  PINF("POSIX-based block-layer component loaded OK.");
}

TEST_F(Simple_pager_test, InstantiatePager)
{
  using namespace Component;
  
  assert(_block);
  IBase * comp = load_component("libcomanche-pagersimple.so",
                                Component::pager_simple_factory);
  assert(comp);
  auto fact = (IPager_factory *) comp->query_interface(IPager_factory::iid());

  _pager = fact->create(12,"testheapid",_block, false);
  ASSERT_TRUE(_pager);
  
  PINF("Pager component loaded OK.");
}



TEST_F(Simple_pager_test, CleanUp)
{
  assert(_block);
  _pager->release_ref();
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
