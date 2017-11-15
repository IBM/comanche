#include <gtest/gtest.h>
#include <string>
#include <list>
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/str_utils.h>
#include <common/logging.h>
#include <common/utils.h>
#include <common/cpu.h>

#include <component/base.h>
#include <api/components.h>
#include <api/block_itf.h>
#include <api/blob_itf.h>

//#define USE_NVME_DEVICE // use real device, POSIX file otherwise

int option_INIT = 0;

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Big_blob_test : public ::testing::Test {

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
  static Component::IRegion_manager * _rm;
  static Component::IBlob * _blob;
  static Component::IBlock_device * _region_bd;
};


Component::IBlock_device *   Big_blob_test::_block;
Component::IBlock_device *   Big_blob_test::_region_bd;
Component::IBlob *           Big_blob_test::_blob;
Component::IRegion_manager * Big_blob_test::_rm;

TEST_F(Big_blob_test, InstantiateBlockDevice)
{
#ifdef USE_NVME_DEVICE
  
  std::string dll_path = getenv("HOME");
  dll_path.append("/comanche/lib/libcomanche-blk.so");
  Component::IBase * comp = Component::load_component(dll_path.c_str(),
                                                      Component::block_nvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  _block = fact->create("./vol-config-local.json");
  assert(_block);
  fact->release_ref();
  PINF("Lower block-layer component loaded OK.");

#else
  
  std::string dll_path = getenv("HOME");
  dll_path.append("/comanche/lib/libcomanche-blkposix.so");
  Component::IBase * comp = Component::load_component(dll_path.c_str(),
                                                      Component::block_posix_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());  
  _block = fact->create("{\"path\":\"blockdev.dat\", \"size_in_blocks\":400000 }");
  assert(_block);
  fact->release_ref();
  PINF("POSIX-based block-layer component loaded OK.");

#endif
}

TEST_F(Big_blob_test, InstantiatePart)
{
  using namespace Component;
  
  assert(_block);
  std::string dll_path = getenv("HOME");
  dll_path.append("/comanche/lib/libcomanche-partregion.so");

  IBase * comp = load_component(dll_path.c_str(),
                                Component::part_region_factory);
  assert(comp);
  IRegion_manager_factory* fact = (IRegion_manager_factory *) comp->query_interface(IRegion_manager_factory::iid());
  assert(fact);

  _rm = fact->open(_block, option_INIT); 

  // bool reused;
  // std::string label = Common::random_string(10);
  // _rm->reuse_or_allocate_region(1, "foobar", label, &reused);
  
  ASSERT_TRUE(_rm);
  fact->release_ref();
  
  PINF("Part-region component loaded OK.");
}

TEST_F(Big_blob_test, InstantiateBlob)
{
  using namespace Component;
  
  assert(_block);
  std::string dll_path = "../libcomanche-blobbig.so";

  IBase * comp = load_component(dll_path.c_str(),
                                Component::blob_big_factory);
  assert(comp);
  IBlob_factory* fact = (IBlob_factory *) comp->query_interface(IBlob_factory::iid());
  assert(fact);

  PINF("blob fact->open flags=%d", option_INIT);

  _blob = fact->open("cocotheclown",
                     "mydb",
                     _rm,
                     GB(1),
                     option_INIT); //IBlob_factory::FLAGS_FORMAT); /* pass in lower-level block device */
  
  ASSERT_TRUE(_blob);
  fact->release_ref();
  
  PINF("Blob component loaded OK.");
}

TEST_F(Big_blob_test, StressBlobCreateDelete)
{

  unsigned total = 0;
  std::list<IBlob::blob_t> handles;

  while(total < 100) {
    handles.push_front(_blob->create(rand() % KB(256)));
    total++;
    /* optionally remove one */
    if(rdtsc() % 4 == 0) {
      IBlob::blob_t handle = handles.back();
      handles.pop_back();
      _blob->erase(handle);
      total--;
    }
  }
  _blob->show_state("*");
  while(!handles.empty()) {
    IBlob::blob_t handle = handles.back();
    handles.pop_back();
    _blob->erase(handle);
  }

  _blob->show_state("*");
}

TEST_F(Big_blob_test, CreateTrim)
{
  IBlob::blob_t handle = _blob->create(KB(32));
  _blob->show_state("*");
  _blob->truncate(handle,KB(24));
  _blob->show_state("*");
  _blob->erase(handle);
}

TEST_F(Big_blob_test, Release)
{ 
  assert(_block);
  _blob->release_ref();
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {

  if(argc > 1 && strcmp(argv[1],"format")==0)
    option_INIT = IBlob_factory::FLAGS_FORMAT;
  
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
