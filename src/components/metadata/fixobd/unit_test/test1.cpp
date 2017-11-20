/* note: we do not include component source, only the API definition */
#include <gtest/gtest.h>
#include <api/components.h>
#include <api/metadata_itf.h>
#include <api/block_itf.h>
#include <core/dpdk.h>

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Fixobd_test : public ::testing::Test {

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
  static Component::IMetadata *      _md;
};

Component::IMetadata * Fixobd_test::_md;
Component::IBlock_device * Fixobd_test::_block;

TEST_F(Fixobd_test, BlockdeviceInstantiate)
{
  Component::IBase * comp = Component::load_component("libcomanche-blkposix.so",
                                                      Component::block_posix_factory);
  assert(comp);

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  std::string config_string;
  config_string = "{\"path\":\"./blockfile.dat\",\"size_in_blocks\":1000}";

  _block = fact->create(config_string);
  assert(_block);
  fact->release_ref();
  PINF("Block-layer component loaded OK (itf=%p)", _block);
}

/*< instantiation helpers */  
IMetadata * component_create_metadata_fixobd(const char * owner,
                                             const char * name,
                                             Component::IBlock_device * block,
                                             int flags)
{
  auto comp = load_component("libcomanche-mdfixobd.so",Component::metadata_fixobd_factory);
  assert(comp);                                                         
  auto fact = (IMetadata_factory *) comp->query_interface(IMetadata_factory::iid());                    
  assert(fact);                                                         
  auto inst = fact->create(owner, name, block, flags);
  assert(inst);
  fact->release_ref();
  return inst;
}

TEST_F(Fixobd_test, Instantiate)
{
  _md = component_create_metadata_fixobd("testowner","myname", _block, 0);
}


TEST_F(Fixobd_test, Release)
{
  ASSERT_TRUE(_md);

  /* release instance */
  _md->release_ref();
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {

  DPDK::eal_init(32, 1, true);
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
