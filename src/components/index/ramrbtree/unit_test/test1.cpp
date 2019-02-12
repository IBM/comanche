/* note: we do not include component source, only the API definition */
#include <gtest/gtest.h>
#include <common/utils.h>
#include <api/components.h>
#include <api/kvindex_itf.h>

#define COUNT 1000000

using namespace Component;
using namespace Common;
using namespace std;

namespace {

// The fixture for testing class Foo.
class KVIndex_test : public ::testing::Test {

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
  static Component::IKVIndex * _kvindex;
};

Component::IKVIndex * KVIndex_test::_kvindex;


TEST_F(KVIndex_test, Instantiate)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-storeindex.so", Component::ramrbtree_factory);

  ASSERT_TRUE(comp);
  IKVIndex_factory * fact = (IKVIndex_factory *) comp->query_interface(IKVIndex_factory::iid());

  _kvindex = fact->create("owner","name");
  
  fact->release_ref();
}

TEST_F(KVIndex_test, Insert)
{
    string keys[COUNT];
    for(int i=0; i<COUNT; i++){
        _kvindex->insert(key);
    }
  PINF("Size: %d", _kvindex->count());
}

TEST_F(KVIndex_test, Get)
{
  std::string a=_kvindex->get(0);
  PINF("Key= %s", a.c_str());
  a=_kvindex->get(1);
  PINF("Key= %s", a.c_str());
}


TEST_F(KVIndex_test, Erase)
{
  _kvindex->erase("MyKey");
}

TEST_F(KVIndex_test, Count){
    PINF("Size: %d", _kvindex->count());
}


} // namespace

int main(int argc, char **argv) {
  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
