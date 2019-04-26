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

#include <common/cpu.h>
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/utils.h>
#include <core/conc_avl_tree.h>
#include <core/dpdk.h>
#include <core/physical_memory.h>
#include <core/uipc.h>
#include <gtest/gtest.h>
#include <unistd.h>
#include <string>

namespace
{
// The fixture for testing class Foo.
class Core_test : public ::testing::Test {
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
};

TEST_F(Core_test, DPDKInit) { DPDK::eal_init(32); }

TEST_F(Core_test, MemoryAllocation) {
  Core::Physical_memory alloc;
  std::vector<Component::io_buffer_t> v_iobs;

  for (unsigned i = 0; i < 100; i++) {
    auto iob = alloc.allocate_io_buffer(rand() % KB(64), 4096,
                                        Component::NUMA_NODE_ANY);
    v_iobs.push_back(iob);
  }

  for (auto& iob : v_iobs) {
    alloc.free_io_buffer(iob);
  }
}

TEST_F(Core_test, ReleasePartition) {
  // template<typename T>
  // int hash(T value);

  // /*!
  //  * Specialization of hash for the int type.
  //  */
  // template<>
  // inline int hash<int>(int value){
  //     return value;
  // }

  using namespace Concurrent;

  struct node_element {
    int i;
    void* p;
  };

  AVL::Tree<struct node_element*> tree;
  for (int i = 0; i < 10; i++) {
    tree.add(i, new struct node_element({i, this}));
  }

  tree.add(9, new struct node_element({9, this}));  // repeat element

  tree.apply_topdown([](key_t key, struct node_element* value) {
    PLOG("key=%d value=%d:%p", key, value->i, value->p);
    delete value;
  });
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
