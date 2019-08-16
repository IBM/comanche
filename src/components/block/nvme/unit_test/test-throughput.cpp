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
  size_t io_size_in_KiB;
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

/** Throughput of 128k write*/
TEST_F(Block_nvme_test, ThroughputBigwrite) {
  using namespace Component;
  size_t io_size=opt.io_size_in_KiB*1024;

  io_buffer_t mem =
      _block->allocate_io_buffer(io_size, 4096, Component::NUMA_NODE_ANY);

  unsigned ITERATIONS = 5000;
  uint64_t tag;

  /* warm up */
  for (unsigned i = 0; i < 100; i++) tag = _block->async_write(mem, 0, i, 1);
  while (!_block->check_completion(tag))
    ; /* we only have to check the last completion */

  cpu_time_t start = rdtsc();

  Component::VOLUME_INFO info;
  _block->get_volume_info(info);

  size_t nr_lbas_per_io = io_size/(info.block_size);
  size_t max_used_lba = GB(128)/(info.block_size); /* Test range*/
  size_t nr_tabs = max_used_lba/nr_lbas_per_io;

#if 0
  uint64_t last_checked = 0;
  uint64_t water_mark = 128;  //(ITERATIONS << 2);

  for (unsigned i = 0; i < ITERATIONS; i++) {
    size_t lba = i*nr_blocks_per_io;
    tag = _block->async_write(mem, 0, lba, nr_blocks_per_io);
    if (tag - last_checked > water_mark) {
      if (_block->check_completion(last_checked + water_mark)) {
        last_checked += water_mark;
      }
    }
  }
  while (!_block->check_completion(tag))
    ;
#else
  for (unsigned i = 0; i < ITERATIONS; i++) {
    size_t lba = (rand() % nr_tabs) * nr_lbas_per_io;
    _block->write(mem, 0, lba, nr_lbas_per_io);
  }
#endif

  cpu_time_t cycles_per_iop = (rdtsc() - start) / (ITERATIONS);
  float cpu_freq_in_mhz = Common::get_rdtsc_frequency_mhz();
  float bw_in_MB =  (io_size)*(cpu_freq_in_mhz) / cycles_per_iop;
  float bw_in_MiB = (1000000.0/MiB(1))* bw_in_MB;
  PINF("[bigwrite]: CPU frequency %.3f Mhz/s", cpu_freq_in_mhz);
  PINF("[bigwrite]: nr_blocks_per_io=%lu, write size=%lu bytes", nr_lbas_per_io, io_size);
  PINF("[bigwrite]: took %ld cycles (%f usec) per IOP", cycles_per_iop,
       cycles_per_iop / (cpu_freq_in_mhz));
  PINF("[bigwrite]: rate: %f KIOPS", (cpu_freq_in_mhz * 1000.0) / cycles_per_iop);
  PINF("[bigwrite]: throughput: %.2f MB/s (%.2f MiB/s)",  bw_in_MB, bw_in_MiB);

  _block->free_io_buffer(mem);
}

TEST_F(Block_nvme_test, ReleaseBlockDevice) {
  assert(_block);
  _block->release_ref();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    PINF("test <pci-address> io_size(4096|65536|131072|262144|2097152)");
    return 0;
  }

  opt.pci = argv[1];
  if(argc == 3){
    opt.io_size_in_KiB = atoi(argv[2]);
  }
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
