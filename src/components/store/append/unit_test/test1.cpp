#include <mcheck.h>
#include <gtest/gtest.h>
#include <string>
#include <list>
#include <set>
#include <chrono>
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/str_utils.h>
#include <common/logging.h>
#include <common/cpu.h>
#include <common/utils.h>
#include <common/dump_utils.h>
#include <common/str_utils.h>
#include <core/avl_malloc.h>
#include <core/dpdk.h>
#include <component/base.h>

#include <api/components.h>
#include <api/block_itf.h>
#include <api/region_itf.h>
#include <api/pmem_itf.h>
#include <api/block_allocator_itf.h>
#include <api/store_itf.h>

#define USE_SPDK_NVME_DEVICE // use SPDK-NVME or POSIX-NVME

std::string device_name_arg;

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Append_store_test : public ::testing::Test {

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
  static Component::IStore   *      _store;
};

Component::IBlock_device * Append_store_test::_block;
Component::IStore * Append_store_test::_store;

TEST_F(Append_store_test, InitDPDK)
{
  DPDK::eal_init(2048);
}

TEST_F(Append_store_test, InstantiateBlockDevice)
{
#ifdef USE_SPDK_NVME_DEVICE
  
  Component::IBase * comp = Component::load_component("libcomanche-blknvme.so",
                                                      Component::block_nvme_factory);

  assert(comp);
  PLOG("Block_device factory loaded OK.");
  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  
  cpu_mask_t cpus;
  cpus.add_core(2);

  _block = fact->create(device_name_arg.c_str(), &cpus);

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
  config_string += "\",\"size_in_blocks\":100000}";
  PLOG("config: %s", config_string.c_str());

  _block = fact->create(config_string);
  assert(_block);
  fact->release_ref();
  PINF("Block-layer component loaded OK (itf=%p)", _block);

#endif
}

TEST_F(Append_store_test, Instantiate)
{
  Component::IBase * comp = Component::load_component("libcomanche-storeappend.so",
                                                      Component::store_append_factory);
  assert(comp);
  PLOG("Append-store factory loaded OK.");

  IStore_factory * fact = (IStore_factory *) comp->query_interface(IStore_factory::iid());

  ASSERT_TRUE(_block);
  _store = fact->create("testowner","teststore",_block, 0);
  ASSERT_TRUE(_store);
  
  fact->release_ref();
}

TEST_F(Append_store_test, CreateEntries)
{
  size_t total_write_size = GB(2);
  std::string val = Common::random_string(MB(1));

  std::vector<std::string> keys;
  for(unsigned i=0;i<total_write_size/MB(1);i++) {
    keys.push_back(Common::random_string(128));
  }

  auto started = std::chrono::high_resolution_clock::now();

  //#pragma omp parallel for shared(keys)
  for(unsigned i=0;i<total_write_size/MB(1);i++) {
    _store->put(keys[i], "metadata", (void*)val.c_str(), val.size());
  }
  _store->flush();
  
  auto done = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();
  PLOG("Duration %f seconds: %lu transactions", ((float)ms)/1000.0, total_write_size/MB(1));
}

TEST_F(Append_store_test, ReleaseBlockDevice)
{
  ASSERT_TRUE(_store);
  ASSERT_TRUE(_block);

  _store->release_ref();
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  mtrace();

  if(argc > 1)
    device_name_arg = argv[1];
  else
    device_name_arg = "86:00.0";
  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  muntrace();
  return r;
}
