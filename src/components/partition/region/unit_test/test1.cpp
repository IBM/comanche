/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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
#include <api/region_itf.h>

//#define USE_NVME_DEVICE // use real device, POSIX file otherwise

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Part_region_test : public ::testing::Test {

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
  static Component::IBlock_device * _region_bd;
};


Component::IBlock_device * Part_region_test::_block;
Component::IBlock_device * Part_region_test::_region_bd;
Component::IRegion_manager * Part_region_test::_rm;

TEST_F(Part_region_test, InstantiateBlockDevice)
{
#ifdef USE_NVME_DEVICE
  
  Component::IBase * comp = Component::load_component("libcomanche-blknvme.so",
                                                      Component::block_nvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  _block = fact->create("8b:00.0");
  assert(_block);
  fact->release_ref();
  PINF("Lower block-layer component loaded OK.");

#else
  
  Component::IBase * comp = Component::load_component("libcomanche-blkposix.so",
                                                      Component::block_posix_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());  
  _block = fact->create("{\"path\":\"blockdev.dat\", \"size_in_blocks\":4000 }");
  assert(_block);
  fact->release_ref();
  PINF("POSIX-based block-layer component loaded OK.");

#endif
}

TEST_F(Part_region_test, InstantiatePart)
{
  using namespace Component;
  
  assert(_block);
  IBase * comp = load_component("libcomanche-partregion.so",
                                Component::part_region_factory);
  assert(comp);
  IRegion_manager_factory* fact = (IRegion_manager_factory *) comp->query_interface(IRegion_manager_factory::iid());
  assert(fact);

  _rm = fact->open(_block,0); // IRegion_manager_factory::FLAGS_FORMAT); /* pass in lower-level block device */

  ASSERT_TRUE(_rm);

  std::string name = Common::random_string(16);
  bool reused = false;
  addr_t vaddr = 0;
  auto region = _rm->reuse_or_allocate_region(rand() % 256,
                                              "dwaddington",
                                              name.c_str(),
                                              vaddr,
                                              reused);

  fact->release_ref();
  
  PINF("Part-region component loaded OK.");
  PINF("Num regions=%ld", _rm->num_regions());
}

#if 0
TEST_F(Part_region_test, PartitionCreateDelete)
{
  static constexpr unsigned NUM_REGIONS = 20;
  
  srand(0);
  std::list<Component::IBlock_device*> regions;
  std::list<std::string> names;
  
  for(unsigned i=0;i<NUM_REGIONS;i++) {

    std::string name = Common::random_string(16);
    names.push_back(name);
    bool reused = false;
    addr_t vaddr = 0;
    auto region = _rm->reuse_or_allocate_region(rand() % 256,
                                                "dwaddington",
                                                name.c_str(),
                                                vaddr,
                                                reused);
    regions.push_back(region);
  }

  PLOG("#regions:%ld", _rm->num_regions());
  ASSERT_TRUE(_rm->num_regions() == NUM_REGIONS);

  for(auto& r: regions) {
    r->release_ref();
  }
  /* delete regions */
  for(auto& n: names) {
    Component::IRegion_manager::REGION_INFO ri;
    ASSERT_TRUE(_rm->delete_region("dwaddington", n));
    ASSERT_FALSE(_rm->get_region_info("dwaddington", n, ri));
  }

  ASSERT_TRUE(_rm->num_regions() == 0);
}
#endif

#if 0
TEST_F(Part_region_test, OpenTestPartition)
{
  bool reused = false;
  addr_t vaddr = 0;
  _region_bd = _rm->reuse_or_allocate_region(1000, "dwaddington","area59", vaddr, reused);
  ASSERT_TRUE(_region_bd);
  PINF("Opened partition");
}
#endif

#if 0
TEST_F(Part_region_test, PartitionIntegrity)
{
  using namespace Component;
  
  io_buffer_t mem = _region_bd->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  void * ptr = _region_bd->virt_addr(mem);
  unsigned ITERATIONS = 10000;

  VOLUME_INFO vinfo;
  _region_bd->get_volume_info(vinfo);
  
  PLOG("Volume Info: size=%ld blocks", vinfo.max_lba);
  PLOG("             block_size=%u", vinfo.block_size);
  PLOG("             name=%s", vinfo.volume_name);

  unsigned BLOCK_COUNT = vinfo.max_lba;

  /* zero blocks first */
  memset(ptr,0,4096);

  uint64_t tag;
  for(unsigned k=0;k < BLOCK_COUNT; k++) {
    _region_bd->write(mem, 0, k, 1);
  }
  
  PLOG("Zeroing complete OK.");

  for(unsigned i=0;i<ITERATIONS;i++) {
    uint64_t lba = rand() % BLOCK_COUNT;
    
    /* read existing content */
    _region_bd->read(mem, 0, lba, 1);
    
    uint64_t * v = (uint64_t*) ptr;
    if(v[0] != 0 && v[0] != lba) {
      PERR("value read from drive = %lx, lba=%ld", *v,lba);
      throw General_exception("bad data!");
    }
    /* TODO: check rest of block? */

    /* write out LBA into first 64bit */
    v[0] = lba;
    _region_bd->write(mem, 0, lba, 1);
  }

  PINF("Integrity check OK.");
}
#endif

TEST_F(Part_region_test, ReleaseBlockDevice)
{
  assert(_block);
  _rm->release_ref();
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
