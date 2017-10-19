#include <gtest/gtest.h>
#include <string>

#include "../src/storage_agent.h"

using namespace std;
using namespace comanche;


namespace {

// The fixture for testing class Foo.
class Storage_agent_test : public ::testing::Test {

 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  Storage_agent_test()
  {
  }

  virtual ~Storage_agent_test() {
    // You can do clean-up work that doesn't throw exceptions here.
  }

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
    //    _sa.start();
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
    //    _sa.shutdown();
  }
  
  // Objects declared here can be used by all tests in the test case for Foo.
  static Storage_agent * _sa;
};

Storage_agent * Storage_agent_test::_sa = NULL;

TEST_F(Storage_agent_test, StorageAgentConstruction) { if(!_sa) _sa = new Storage_agent("test-vol-config.json"); }
TEST_F(Storage_agent_test, Waiting) { while(1) sleep(1); }

// // Tests that the Foo::Bar() method does Abc.
// TEST_F(Storage_agent_test, MethodBarDoesAbc) {
//   const string input_filepath = "this/package/testdata/myinputfile.dat";
//   const string output_filepath = "this/package/testdata/myoutputfile.dat";
//   Foo f;
//   EXPECT_EQ(0, f.Bar(input_filepath, output_filepath));
// }

// // Tests that Foo does Xyz.
// TEST_F(Storage_agent_test, DoesXyz) {
//   // Exercises the Xyz feature of Foo.
// }

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
