/* note: we do not include component source, only the API definition */
#include <gtest/gtest.h>
#include <api/components.h>
#include <api/cluster_itf.h>
#include <common/utils.h>

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Zyre_test : public ::testing::Test {

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
  static Component::ICluster * _zyre;
};

bool client;
Component::ICluster * Zyre_test::_zyre;

TEST_F(Zyre_test, Instantiate)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-zyre.so",
                                                      Component::cluster_zyre_factory);

  ASSERT_TRUE(comp);
  ICluster_factory * fact = (ICluster_factory *) comp->query_interface(ICluster_factory::iid());

  _zyre = fact->create("bigboy");
  
  fact->release_ref();
}


TEST_F(Zyre_test, Cleanup)
{
  _zyre->release_ref();
}



} // namespace

int main(int argc, char **argv) {

  if(argc < 2) {
    PINF("zyre-test1");
    return -1;
  }

  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
