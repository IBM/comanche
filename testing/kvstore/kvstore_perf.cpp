/* note: we do not include component source, only the API definition */
#include <gtest/gtest.h>
#include <common/utils.h>
#include <api/components.h>
#include <api/kvstore_itf.h>

using namespace Component;

static Component::IKVStore::pool_t pool;

namespace {

// The fixture for testing class Foo.
class KVStore_test : public ::testing::Test {

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
  static Component::IKVStore * _kvstore;
};

Component::IKVStore * KVStore_test::_kvstore;

#define PATH "/mnt/pmem0/"

TEST_F(KVStore_test, Instantiate)
{
  /* create object instance through factory */
  //  Component::IBase * comp = Component::load_component("libcomanche-storefile.so", Component::filestore_factory);
  //  Component::IBase * comp = Component::load_component("libcomanche-pmstore.so", Component::pmstore_factory);
  Component::IBase * comp = Component::load_component("/home/danielwaddington/comanche/build/comanche-restricted/src/components/pmstore/libcomanche-pmstore.so", Component::pmstore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  _kvstore = fact->create("owner","name");
  
  fact->release_ref();
}


TEST_F(KVStore_test, OpenPool)
{
  ASSERT_TRUE(_kvstore);
  try {
    pool = _kvstore->create_pool(PATH, "test1.rksdb", GB(4));
  }
  catch(...) {
    pool = _kvstore->open_pool(PATH, "test1.rksdb");
  }
  ASSERT_TRUE(pool != 0);
}


TEST_F(KVStore_test, BasicPut)
{
  ASSERT_TRUE(pool);
  std::string key = "MyKey";
  std::string value = "Hello world!";
  //  value.resize(value.length()+1); /* append /0 */
  value.resize(MB(8));
    
  _kvstore->put(pool, key, value.c_str(), value.length());
}

TEST_F(KVStore_test, BasicGet)
{
  std::string key = "MyKey";

  void * value = nullptr;
  size_t value_len = 0;
  _kvstore->get(pool, key, value, value_len);
  PINF("Value=(%.50s) %lu", ((char*)value), value_len);
}


TEST_F(KVStore_test, BasicRemove)
{
  _kvstore->erase(pool, "MyKey");
}

TEST_F(KVStore_test, ClosePool)
{
  _kvstore->close_pool(pool);
}

#if 0
TEST_F(KVStore_test, ReopenPool)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->open_pool(PATH, "test1.rksdb");
  ASSERT_TRUE(pool != 0);
}

TEST_F(KVStore_test, DeletePool)
{
  _kvstore->delete_pool(pool);
}
#endif


} // namespace

int main(int argc, char **argv) {
  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
