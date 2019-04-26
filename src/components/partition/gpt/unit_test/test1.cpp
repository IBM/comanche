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
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/cpu.h>

#include <component/base.h>
#include <api/components.h>
#include <api/block_itf.h>
#include <api/partition_itf.h>

//#define USE_NVME_DEVICE

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Part_gpt_test : public ::testing::Test {

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
  static Component::IPartitioned_device * _pd;
  static Component::IBlock_device * _scoped_bd;
};


Component::IBlock_device * Part_gpt_test::_block;
Component::IBlock_device * Part_gpt_test::_scoped_bd;
Component::IPartitioned_device * Part_gpt_test::_pd;



TEST_F(Part_gpt_test, InstantiateBlockDevice)
{
#ifdef USE_NVME_DEVICE
  
  Component::IBase * comp = Component::load_component("libcomanche-blknvme.so",
                                                      Component::block_nvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  cpu_mask_t cpus;
  cpus.add_core(2);

  _block = fact->create("86:00.0", &cpus);
  assert(_block);
  fact->release_ref();
  PINF("Lower block-layer component loaded OK.");

#else
  Component::IBase * comp = Component::load_component("libcomanche-blkposix.so",
                                                      Component::block_posix_factory);
  assert(comp);

  unlink("./blockfile.dat");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  std::string config_string;
  config_string = "{\"path\":\"./blockfile.dat\",\"size_in_blocks\":100}";
  _block = fact->create(config_string);
  assert(_block);
  fact->release_ref();
  PINF("Block-layer component loaded OK (itf=%p)", _block);
#endif
}

TEST_F(Part_gpt_test, InstantiatePart)
{
  using namespace Component;
  
  assert(_block);
  IBase * comp = load_component("libcomanche-partgpt.so",
                                Component::part_gpt_factory);
  assert(comp);
  IPartitioned_device_factory* fact = (IPartitioned_device_factory *) comp->query_interface(IPartitioned_device_factory::iid());
  assert(fact);

  _pd = fact->create(_block); /* pass in lower-level block device */
  fact->release_ref();
  
  PINF("Part-GPT component loaded OK.");
}

TEST_F(Part_gpt_test, OpenPartition)
{
  _scoped_bd = _pd->open_partition(0);
}

TEST_F(Part_gpt_test, PartitionIntegrity)
{
  using namespace Component;
  
  io_buffer_t mem = _scoped_bd->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  void * ptr = _scoped_bd->virt_addr(mem);
  unsigned ITERATIONS = 10000;

  VOLUME_INFO vinfo;
  _scoped_bd->get_volume_info(vinfo);
  
  PLOG("Volume Info: size=%ld blocks", vinfo.block_count);
  PLOG("             block_size=%u", vinfo.block_size);
  PLOG("             name=%s", vinfo.volume_name);

  unsigned BLOCK_COUNT = vinfo.block_count;

  /* zero blocks first */
  memset(ptr,0,4096);

  uint64_t tag;
  for(unsigned k=0;k < BLOCK_COUNT; k++) {
    tag = _scoped_bd->async_write(mem, 0, k, 1);
  }
  while(!_scoped_bd->check_completion(tag)) usleep(1000);
  
  PLOG("Zeroing complete OK.");

  for(unsigned i=0;i<ITERATIONS;i++) {
    uint64_t lba = rand() % BLOCK_COUNT;
    
    /* read existing content */
    tag = _scoped_bd->async_read(mem, 0, lba, 1);
    while(!_scoped_bd->check_completion(tag)) usleep(1000);
    
    uint64_t * v = (uint64_t*) ptr;
    if(v[0] != 0 && v[0] != lba) {
      PERR("value read from drive = %lx, lba=%ld", *v,lba);
      throw General_exception("bad data!");
    }
    /* TODO: check rest of block? */

    /* write out LBA into first 64bit */
    v[0] = lba;
    tag = _scoped_bd->async_write(mem, 0, lba, 1);
    while(!_scoped_bd->check_completion(tag)) usleep(1000);
  }

  PINF("Integrity check OK.");
}


TEST_F(Part_gpt_test, ReleasePartition)
{
  _pd->release_partition(_scoped_bd);
}

TEST_F(Part_gpt_test, ReleaseBlockDevice)
{
  assert(_block);
  _pd->release_ref();
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
