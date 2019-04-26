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
#include <core/postbox.h>
#include <core/uipc.h>
#include <gtest/gtest.h>
#include <unistd.h>
#include <string>

#include <core/rlf_bitmap.h>

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

#if 0
TEST_F(Core_test, Bitmap)
{
  size_t n_elements = 128;
  size_t size = Core::Relocatable_LF_bitmap::required_memory_size(n_elements);
  void * ptr = malloc(size);
  PLOG("ptr=%p", ptr);
  Core::Relocatable_LF_bitmap * slab = new (ptr) Core::Relocatable_LF_bitmap(size,n_elements);

  for(unsigned i=0;i<128;i++) {
    auto r = slab->allocate();
    PLOG("r=%u", r);
  }
  PLOG("next should fail..");
  ASSERT_ANY_THROW(auto r_none = slab->allocate());

  for(unsigned i=0;i<64;i++) {
    slab->free(i);
  }

  for(unsigned i=0;i<64;i++) {
    auto r = slab->allocate();
    PLOG("r2=%u", r);
  }

  ASSERT_ANY_THROW(auto r_none = slab->allocate());
}
#endif

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
