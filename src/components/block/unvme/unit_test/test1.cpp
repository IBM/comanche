/*
   Copyright [2017] [IBM Corporation]

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
#include <api/fs_itf.h>

using namespace Component;

struct
{
  std::string pci;
} opt;

namespace {

// The fixture for testing class Foo.
class Block_unvme_test : public ::testing::Test {

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
};


Component::IBlock_device * Block_unvme_test::_block;

TEST_F(Block_unvme_test, InstantiateBlockDevice)
{
  std::string dll_path = getenv("HOME");
  dll_path.append("/comanche/lib/libcomanche-blkunvme.so");
  Component::IBase * comp = Component::load_component(dll_path.c_str(),
                                                      Component::block_unvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  _block = fact->create(opt.pci.c_str());
  
  assert(_block);
  fact->release_ref();
  PINF("uNVME-based block-layer component loaded OK.");
}

TEST_F(Block_unvme_test, PartitionIntegrity)
{
  using namespace Component;
  
  io_buffer_t mem = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  void * ptr = _block->virt_addr(mem);
  unsigned ITERATIONS = 100000;

  VOLUME_INFO vinfo;
  _block->get_volume_info(vinfo);
  
  PLOG("Volume Info: size=%ld blocks", vinfo.block_count);
  PLOG("             block_size=%u", vinfo.block_size);
  PLOG("             name=%s", vinfo.volume_name);

  //  unsigned BLOCK_COUNT = vinfo.max_lba;
  unsigned BLOCK_COUNT = 20000; // limit size

  /* zero blocks first */
  memset(ptr,0,4096);

  uint64_t tag;

  _block->write(mem, 0, 0, 1);
  PLOG("First block write OK.");
  
  for(unsigned k=0;k < BLOCK_COUNT; k++) {
    _block->write(mem, 0, k, 1);
  }  
  
  PLOG("Zeroing complete OK.");

  for(unsigned i=0;i<ITERATIONS;i++) {
    uint64_t lba = rand() % BLOCK_COUNT;
    
    /* read existing content */
    tag = _block->async_read(mem, 0, lba, 1);
    while(!_block->check_completion(tag));
    
    uint64_t * v = (uint64_t*) ptr;
    if(v[0] != 0 && v[0] != lba) {
      PERR("value read from drive = %lx, lba=%ld", *v,lba);
      throw General_exception("bad data!");
    }
    /* TODO: check rest of block? */

    /* write out LBA into first 64bit */
    v[0] = lba;
    tag = _block->async_write(mem, 0, lba, 1);
    while(!_block->check_completion(tag));
  }

  PINF("Integrity check OK.");
}

#if 1
TEST_F(Block_unvme_test, WriteThroughput)
{
  using namespace Component;
  
  set_cpu_affinity(1UL << 2);

  sleep(1);
  
  io_buffer_t mem = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  
  unsigned ITERATIONS = 1000000;
  uint64_t tags[ITERATIONS];

  /* warm up */
  for(unsigned i=0;i<100;i++) 
    tags[i] = _block->async_write(mem, 0, i, 1);
  while(!_block->check_completion(tags[99])); /* we only have to check the last completion */

  cpu_time_t start = rdtsc();

  for(unsigned i=0;i<ITERATIONS;i++) {
    tags[i] = _block->async_write(mem, 0, i, 1);
    //    PLOG("issued tag: %ld", tags[i]);
  }
  while(!_block->check_completion(tags[ITERATIONS-1]));

  cpu_time_t cycles_per_iop = (rdtsc() - start)/(ITERATIONS);
  PINF("took %ld cycles (%f usec) per IOP", cycles_per_iop,  cycles_per_iop / 2400.0f);
  PINF("rate: %f KIOPS", (2400.0 * 1000.0)/cycles_per_iop);

  _block->free_io_buffer(mem);
}
#endif


#if 1
TEST_F(Block_unvme_test, WriteLatency)
{
  set_cpu_affinity(1UL << 2);

  sleep(1);
  
  io_buffer_t mem = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);

  unsigned ITERATIONS = 1000;
  uint64_t tags[ITERATIONS];

  /* warm up */
  for(unsigned i=0;i<100;i++) 
    tags[i] = _block->async_write(mem, 0, i, 1);
  while(!_block->check_completion(tags[99])); /* we only have to check the last completion */

  for(unsigned i=0;i<ITERATIONS;i++) {
    cpu_time_t start = rdtsc();
    _block->write(mem, 0, i, 1);
    cpu_time_t cycles_for_iop = rdtsc() - start;
    PINF("took %ld cycles (%f usec) per IOP", cycles_for_iop,  cycles_for_iop / 2400.0f);
  }

  _block->free_io_buffer(mem);
}
#endif


TEST_F(Block_unvme_test, ReleaseBlockDevice)
{
  assert(_block);
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  if(argc!=2) {
    PINF("test <pci-address>");
    return 0;
  }

  opt.pci = argv[1];
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
