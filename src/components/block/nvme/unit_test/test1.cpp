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
#include <common/cpu.h>
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <gtest/gtest.h>
#include <string>

#include <api/block_itf.h>
#include <api/components.h>
#include <api/fs_itf.h>
#include <component/base.h>

using namespace Component;

struct {
  std::string pci;
} opt;

namespace
{
// The fixture for testing class Foo.
class Block_nvme_test : public ::testing::Test {
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
  static Component::IBlock_device *_block;
};

Component::IBlock_device *Block_nvme_test::_block;

TEST_F(Block_nvme_test, InstantiateBlockDevice) {
  Component::IBase *comp = Component::load_component(
      "libcomanche-blknvme.so", Component::block_nvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory *fact = (IBlock_device_factory *) comp->query_interface(
      IBlock_device_factory::iid());
  cpu_mask_t cpus;
  cpus.add_core(2);
  //  cpus.add_core(25);

  _block = fact->create(opt.pci.c_str(), &cpus);

  assert(_block);
  fact->release_ref();
  PINF("nvme-based block-layer component loaded OK.");
}

#if 0
TEST_F(Block_nvme_test, PartitionIntegrity)
{
  using namespace Component;
  
  io_buffer_t mem = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  void * ptr = _block->virt_addr(mem);
  unsigned ITERATIONS = 1000000;

  VOLUME_INFO vinfo;
  _block->get_volume_info(vinfo);
  
  PLOG("Volume Info: size=%ld blocks", vinfo.max_lba);
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

  srand(rdtsc());
  
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

    if(i % 10000 == 0) PLOG("iteration:%u",i);
  }

  PINF("Integrity check OK.");
}
#endif

#if 1
TEST_F(Block_nvme_test, Throughput) {
  using namespace Component;

  io_buffer_t mem =
      _block->allocate_io_buffer(4096, 4096, Component::NUMA_NODE_ANY);

  unsigned ITERATIONS = 1000000;
  uint64_t tag;

  /* warm up */
  for (unsigned i = 0; i < 100; i++) tag = _block->async_write(mem, 0, i, 1);
  while (!_block->check_completion(tag))
    ; /* we only have to check the last completion */

  cpu_time_t start = rdtsc();

  uint64_t last_checked = 0;
  uint64_t water_mark = 2048;  //(ITERATIONS << 2);

  for (unsigned i = 0; i < ITERATIONS; i++) {
    tag = _block->async_write(mem, 0, i, 1);
    if (tag - last_checked > water_mark) {
      if (_block->check_completion(last_checked + water_mark)) {
        last_checked += water_mark;
      }
    }
  }
  while (!_block->check_completion(tag))
    ;

  cpu_time_t cycles_per_iop = (rdtsc() - start) / (ITERATIONS);
  PINF("[async write]: took %ld cycles (%f usec) per IOP", cycles_per_iop,
       cycles_per_iop / 2400.0f);
  PINF("[async write]: rate: %f KIOPS", (2400.0 * 1000.0) / cycles_per_iop);

  /* also check the the sync read */
  start = rdtsc();

  for (unsigned i = 0; i < ITERATIONS; i++) {
    _block->read(mem, 0, i, 1);
  }
  while (!_block->check_completion(tag))
    ;

  cycles_per_iop = (rdtsc() - start) / (ITERATIONS);
  PINF("[sync_read]: took %ld cycles (%f usec) per IOP", cycles_per_iop,
       cycles_per_iop / 2400.0f);
  PINF("[sync_read]: rate: %f KIOPS", (2400.0 * 1000.0) / cycles_per_iop);

  _block->free_io_buffer(mem);
}
#endif

#if 0
TEST_F(Block_nvme_test, WriteLatencyChecked)
{
  io_buffer_t mem = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);

  unsigned ITERATIONS = 1000;

  srand(rdtsc());
  
  /* warm up */
  // for(unsigned i=0;i<10000;i++) 
  //   tag = _block->async_write(mem, 0, i, 1);
  // while(!_block->check_completion(tag)); /* we only have to check the last completion */
  
  uint64_t * ptr = (uint64_t*) _block->virt_addr(mem);
  
  for(unsigned i=0;i<ITERATIONS;i++) {
    cpu_time_t start = rdtsc();
    ptr[0] = start;
    _block->write(mem, 0, i, 1);

    cpu_time_t cycles_for_iop = rdtsc() - start;
    PINF("checked write latency took %ld cycles (%f usec) per IOP", cycles_for_iop,  cycles_for_iop / 2400.0f);

    ptr[0] = 0;
    _block->read(mem, 0, i, 1);
    ASSERT_TRUE(ptr[0] == start);
  }

  _block->free_io_buffer(mem);
}
#endif

#if 0
TEST_F(Block_nvme_test, CheckAllWriteThroughput)
{
  using namespace Component;
  
  io_buffer_t mem = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  
  unsigned ITERATIONS = 10000;
  uint64_t tag;

  /* warm up */
  cpu_time_t start = rdtsc();

  uint64_t tags[ITERATIONS];
  for(unsigned i=0;i<ITERATIONS;i++) {
    tags[i] = _block->async_write(mem, 0, i, 1);
    //    PLOG("issued tag: %ld", tags[i]);
    if(tags[i] % 4096) _block->check_completion(tags[i]);
  }
  for(unsigned i=0;i<ITERATIONS;i++) {
    while(!_block->check_completion(tags[i]));
  }

  cpu_time_t cycles_per_iop = (rdtsc() - start)/(ITERATIONS);
  PINF("check all took %ld cycles (%f usec) per IOP", cycles_per_iop,  cycles_per_iop / 2400.0f);
  PINF("rate: %f KIOPS", (2400.0 * 1000.0)/cycles_per_iop);

  _block->free_io_buffer(mem);
}
#endif

#if 0
TEST_F(Block_nvme_test, WriteLatency)
{
  io_buffer_t mem = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);

  unsigned ITERATIONS = 1000;
  uint64_t tag;

  /* warm up */
  for(unsigned i=0;i<10000;i++) 
    tag = _block->async_write(mem, 0, i, 1);
  while(!_block->check_completion(tag)); /* we only have to check the last completion */
  

  for(unsigned i=0;i<ITERATIONS;i++) {
    cpu_time_t start = rdtsc();
    _block->write(mem, 0, i, 1);
    cpu_time_t cycles_for_iop = rdtsc() - start;
    PINF("write latency took %ld cycles (%f usec) per IOP", cycles_for_iop,  cycles_for_iop / 2400.0f);
  }

  _block->free_io_buffer(mem);
}
#endif

#if 0
TEST_F(Block_nvme_test, SharedWork)
{
  _block->attach_work(0,[](void* arg) { PLOG("shared work (%p)", arg); }, this);
  io_buffer_t mem = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);

  unsigned ITERATIONS = 20;
  uint64_t tag;

  for(unsigned i=0;i<ITERATIONS;i++) {
    cpu_time_t start = rdtsc();
    _block->write(mem, 0, i, 1);
    cpu_time_t cycles_for_iop = rdtsc() - start;
    PINF("write latency took %ld cycles (%f usec) per IOP", cycles_for_iop,  cycles_for_iop / 2400.0f);
  }

  _block->free_io_buffer(mem);
}
#endif

TEST_F(Block_nvme_test, ReleaseBlockDevice) {
  assert(_block);
  _block->release_ref();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    PINF("test <pci-address>");
    return 0;
  }

  opt.pci = argv[1];
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
