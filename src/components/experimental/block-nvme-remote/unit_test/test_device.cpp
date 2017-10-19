#include <gtest/gtest.h>
#include <string>
#include <common/cycles.h>
#include <nvme_device.h>
using namespace std;

static char device_name[256];

namespace {

// The fixture for testing class Foo.
class Device_test : public ::testing::Test {

 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  Device_test() 
  {
    // // You can do set-up work for each test here.
    if(!_device)
      _device = new Nvme_device(device_name);
  }

  virtual ~Device_test() {
    // You can do clean-up work that doesn't throw exceptions here.
  }

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
  static Nvme_device* _device;
};


Nvme_device * Device_test::_device = NULL;

TEST_F(Device_test, ReadBlock)
{
  for(unsigned i=0;i<1;i++) {
    _device->raw_read(i); /* read blocks 0 through 9 */
  }
}

TEST_F(Device_test, Shutdown) {  delete _device; }

// // Tests that the Foo::Bar() method does Abc.
// TEST_F(Device_test, MethodBarDoesAbc) {
//   const string input_filepath = "this/package/testdata/myinputfile.dat";
//   const string output_filepath = "this/package/testdata/myoutputfile.dat";
//   Foo f;
//   EXPECT_EQ(0, f.Bar(input_filepath, output_filepath));
// }

// // Tests that Foo does Xyz.
// TEST_F(Device_test, DoesXyz) {
//   // Exercises the Xyz feature of Foo.
// }

}  // namespace

int main(int argc, char **argv) {
  assert(argc > 1);
  strcpy(device_name, argv[1]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
