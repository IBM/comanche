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
  std::string pci2;
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
  static Component::IBlock_device *_block0;
  static Component::IBlock_device *_block1;
};

Component::IBlock_device *Block_nvme_test::_block0;
Component::IBlock_device *Block_nvme_test::_block1;

TEST_F(Block_nvme_test, InstantiateBlockDevice0) {
  Component::IBase *comp = Component::load_component(
      "libcomanche-blknvme.so", Component::block_nvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory *fact = (IBlock_device_factory *) comp->query_interface(
      IBlock_device_factory::iid());
  cpu_mask_t cpus;
  cpus.add_core(24);

  _block0 = fact->create(opt.pci.c_str(), &cpus);

  assert(_block0);
  fact->release_ref();
  PINF("nvme-based block-layer (0) component loaded OK.");
}

TEST_F(Block_nvme_test, InstantiateBlockDevice1) {
  Component::IBase *comp = Component::load_component(
      "libcomanche-blknvme.so", Component::block_nvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory *fact = (IBlock_device_factory *) comp->query_interface(
      IBlock_device_factory::iid());
  cpu_mask_t cpus;
  cpus.add_core(25);

  _block1 = fact->create(opt.pci2.c_str(), &cpus);

  assert(_block1);
  fact->release_ref();
  PINF("nvme-based block-layer (1) component loaded OK.");
}

TEST_F(Block_nvme_test, ReleaseBlockDevice) {
  _block0->release_ref();
  _block1->release_ref();
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    PINF("test <pci-address> <pci-address-2>");
    return 0;
  }

  opt.pci = argv[1];
  opt.pci2 = argv[2];

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
