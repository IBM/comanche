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
#include <common/rand.h>
#include <component/base.h>
#include <core/poller.h>
#include <api/components.h>
#include <api/block_itf.h>
#include <api/raid_itf.h>
#include <api/fs_itf.h>

#define IO_CORE 24
#define QUEUE_ID IO_CORE

using namespace Component;

struct
{
  std::vector<std::string> pci_devices;
} opt;

namespace {

// The fixture for testing class Foo.
class Block_raid_test : public ::testing::Test {

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
  static std::vector<Component::IBlock_device *> bd_vector;
  static Component::IRaid * raid;
};


std::vector<Component::IBlock_device *> Block_raid_test::bd_vector;
Component::IRaid * Block_raid_test::raid;

Core::Poller *poller;

TEST_F(Block_raid_test, InstantiateBlockDevices)
{
  Component::IBase * comp = Component::load_component("libcomanche-blknvme.so",
                                                      Component::block_nvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  cpu_mask_t m;
  m.add_core(IO_CORE);
  
  poller = new Core::Poller(m);

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());

  /* instantiate block devices */
  for(auto& pci: opt.pci_devices) {
    PLOG("creating block device on %s", pci.c_str());
    auto block_device = fact->create(pci.c_str(), nullptr, poller);
    ASSERT_FALSE(block_device==nullptr);
    bd_vector.push_back(block_device);
  }
  
  fact->release_ref();
  PINF("uNVME-based block-layer component loaded OK.");
}

TEST_F(Block_raid_test, InstantiateRaidComponent)
{
  Component::IBase * comp = Component::load_component("libcomanche-raid.so",
                                                      Component::block_raid);
  assert(comp);

  raid = (IRaid *) comp->query_interface(IRaid::iid());
}

TEST_F(Block_raid_test, ConfigureRaidComponent)
{
  for(auto& bd : bd_vector)
    raid->add_device(bd);

  raid->configure("{\"raidlevel\" : 0 }");
}

unsigned ITERATIONS = 1000000;

TEST_F(Block_raid_test, TestReadThroughput)
{  
  io_buffer_t mem = raid->allocate_io_buffer(4096,
                                             4096,
                                             Component::NUMA_NODE_ANY);
  uint64_t tag;
  
  cpu_time_t start = rdtsc();

  uint64_t last_checked = 0;
  uint64_t water_mark = 2048; //(ITERATIONS << 2);
  
  for(unsigned i=0;i<ITERATIONS;i++) {
    tag = raid->async_read(mem, 0, i, 1, QUEUE_ID);    
    if((raid->gwid_to_seq(tag) - last_checked) > water_mark) {
      if(raid->check_completion(last_checked+water_mark, QUEUE_ID)) {
        last_checked+=water_mark;
      }
    }
  }
  while(!raid->check_completion(raid->gwid_to_seq(tag), QUEUE_ID)) cpu_relax();


  cpu_time_t cycles_per_iop = (rdtsc() - start)/(ITERATIONS);
  PINF("took %ld cycles (%f usec) per IOP", cycles_per_iop,  cycles_per_iop / 2400.0f);
  PINF("READ rate: %f KIOPS", (2400.0 * 1000.0)/cycles_per_iop);

  raid->free_io_buffer(mem);

}

TEST_F(Block_raid_test, TestReadThroughput2)
{  
  io_buffer_t mem = raid->allocate_io_buffer(4096,
                                             4096,
                                             Component::NUMA_NODE_ANY);
  uint64_t tag;
  
  cpu_time_t start = rdtsc();

  uint64_t last_checked = 0;
  uint64_t water_mark = 2048; //(ITERATIONS << 2);
  
  for(unsigned i=0;i<ITERATIONS;i++) {
    tag = raid->async_read(mem, 0, i, 1, QUEUE_ID);    
    if((raid->gwid_to_seq(tag) - last_checked) > water_mark) {
      if(raid->check_completion(last_checked+water_mark, QUEUE_ID)) {
        last_checked+=water_mark;
      }
    }
  }
  while(!raid->check_completion(raid->gwid_to_seq(tag), QUEUE_ID)) cpu_relax();


  cpu_time_t cycles_per_iop = (rdtsc() - start)/(ITERATIONS);
  PINF("took %ld cycles (%f usec) per IOP", cycles_per_iop,  cycles_per_iop / 2400.0f);
  PINF("READ rate: %f KIOPS", (2400.0 * 1000.0)/cycles_per_iop);

  raid->free_io_buffer(mem);

}

TEST_F(Block_raid_test, TestRandomReadThroughput)
{  
  io_buffer_t mem = raid->allocate_io_buffer(4096,
                                             4096,
                                             Component::NUMA_NODE_ANY);
  uint64_t tag;
  
  cpu_time_t start = rdtsc();

  uint64_t last_checked = 0;
  uint64_t water_mark = 2048; //(ITERATIONS << 2);
  VOLUME_INFO vi;
  raid->get_volume_info(vi);
  
  for(unsigned i=0;i<ITERATIONS;i++) {
    tag = raid->async_read(mem, 0, genrand64_int64() % vi.max_lba, 1, QUEUE_ID);    
    if((raid->gwid_to_seq(tag) - last_checked) > water_mark) {
      if(raid->check_completion(last_checked+water_mark, QUEUE_ID)) {
        last_checked+=water_mark;
      }
    }
  }
  while(!raid->check_completion(raid->gwid_to_seq(tag), QUEUE_ID)) cpu_relax();


  cpu_time_t cycles_per_iop = (rdtsc() - start)/(ITERATIONS);
  PINF("took %ld cycles (%f usec) per IOP", cycles_per_iop,  cycles_per_iop / 2400.0f);
  PINF("RANDOM READ rate: %f KIOPS", (2400.0 * 1000.0)/cycles_per_iop);

  raid->free_io_buffer(mem);
}

TEST_F(Block_raid_test, TestWriteThroughput)
{  
  io_buffer_t mem = raid->allocate_io_buffer(4096,
                                             4096,
                                             Component::NUMA_NODE_ANY);
  cpu_time_t start = rdtsc();
  uint64_t tag;
  uint64_t last_checked = 0;
  uint64_t water_mark = 2048; //(ITERATIONS << 2);
  
  for(unsigned i=0;i<ITERATIONS;i++) {
    tag = raid->async_write(mem, 0, i, 1, QUEUE_ID);    
    if((raid->gwid_to_seq(tag) - last_checked) > water_mark) {
      if(raid->check_completion(last_checked+water_mark, QUEUE_ID)) {
        last_checked+=water_mark;
      }
    }
  }
  while(!raid->check_completion(raid->gwid_to_seq(tag), QUEUE_ID)) cpu_relax();


  cpu_time_t cycles_per_iop = (rdtsc() - start)/(ITERATIONS);
  PINF("took %ld cycles (%f usec) per IOP", cycles_per_iop,  cycles_per_iop / 2400.0f);
  PINF("WRITE rate: %f KIOPS", (2400.0 * 1000.0)/cycles_per_iop);

  raid->free_io_buffer(mem);
}

TEST_F(Block_raid_test, TestRandomWriteThroughput)
{  
  io_buffer_t mem = raid->allocate_io_buffer(4096,
                                             4096,
                                             Component::NUMA_NODE_ANY);
  cpu_time_t start = rdtsc();
  uint64_t tag;
  uint64_t last_checked = 0;
  uint64_t water_mark = 2048; //(ITERATIONS << 2);
  VOLUME_INFO vi;
  raid->get_volume_info(vi);
  
  for(unsigned i=0;i<ITERATIONS;i++) {
    tag = raid->async_write(mem, 0, genrand64_int64() % vi.max_lba, 1, QUEUE_ID);    
    if((raid->gwid_to_seq(tag) - last_checked) > water_mark) {
      if(raid->check_completion(last_checked+water_mark, QUEUE_ID)) {
        last_checked+=water_mark;
      }
    }
  }
  while(!raid->check_completion(raid->gwid_to_seq(tag), QUEUE_ID)) cpu_relax();


  cpu_time_t cycles_per_iop = (rdtsc() - start)/(ITERATIONS);
  PINF("took %ld cycles (%f usec) per IOP", cycles_per_iop,  cycles_per_iop / 2400.0f);
  PINF("RAND WRITE rate: %f KIOPS", (2400.0 * 1000.0)/cycles_per_iop);

  raid->free_io_buffer(mem);

}

#if 0
TEST_F(Block_raid_test, WriteLatency)
{
  io_buffer_t mem = raid->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);

  unsigned ITERATIONS = 1000;
  uint64_t tag;

  /* warm up */
  for(unsigned i=0;i<10000;i++) 
    tag = raid->async_write(mem, 0, i, 1, QUEUE_ID);
  while(!raid->check_completion(raid->gwid_to_seq(tag), QUEUE_ID)); /* we only have to check the last completion */
  

  for(unsigned i=0;i<ITERATIONS;i++) {
    cpu_time_t start = rdtsc();
    raid->write(mem, 0, i, 1, QUEUE_ID);
    cpu_time_t cycles_for_iop = rdtsc() - start;
    PINF("write latency took %ld cycles (%f usec) per IOP", cycles_for_iop,  cycles_for_iop / 2400.0f);
  }

  raid->free_io_buffer(mem);
}
#endif

TEST_F(Block_raid_test, ReleaseBlockDevice)
{
  ASSERT_TRUE(raid);
  
  raid->release_ref();
  for(auto& bd : bd_vector) {
    ASSERT_TRUE(bd);
    bd->release_ref();
  }

  delete poller;
}


} // namespace

int main(int argc, char **argv) {
  if(argc < 2) {
    PINF("test <pci-address> <pci-address> ....");
    return 0;
  }

  for(unsigned i=1;i<argc;i++) {
    assert(argv[i] != nullptr);
    opt.pci_devices.push_back(argv[i]);
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
