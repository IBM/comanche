#include <gtest/gtest.h>
#include <string>
#include <unistd.h>
#include <common/cycles.h>
#include <common/utils.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/cpu.h>
#include <core/conc_avl_tree.h>
#include <core/dpdk.h>
#include <core/physical_memory.h>
#include <core/uipc.h>

namespace {

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

bool client_side = false;

TEST_F(Core_test, UIPC)
{
  
  if(!client_side) {
    Core::UIPC::Shared_memory sm("/tmp/foobar", 4096);
  }
  else {
    Core::UIPC::Shared_memory sm("/tmp/foobar");
  }
}


} // namespace

int main(int argc, char **argv) {

  if(argc > 1) client_side = true;
  
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
