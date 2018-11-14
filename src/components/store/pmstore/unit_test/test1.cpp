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

#define PMEM_PATH "/mnt/pmem0/pool/0/"
//#define PMEM_PATH "/dev/pmem0"


TEST_F(KVStore_test, Instantiate)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-pmstore.so",
                                                      Component::pmstore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  _kvstore = fact->create("owner","name");
  
  fact->release_ref();
}

TEST_F(KVStore_test, OpenPool)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->create_pool(PMEM_PATH, "test-pmstore.pool", MB(128));
  ASSERT_TRUE(pool > 0);
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

// TEST_F(KVStore_test, BasicGetRef)
// {
//   std::string key = "MyKey";

//   void * value = nullptr;
//   size_t value_len = 0;
//   _kvstore->get_reference(pool, key, value, value_len);
//   PINF("Ref Value=(%.50s) %lu", ((char*)value), value_len);
//   _kvstore->release_reference(pool, value);
// }


// TEST_F(KVStore_test, BasicMap)
// {
//   _kvstore->map(pool,[](const std::string& key,
//                         const void * value,
//                         const size_t value_len) -> int
//                 {
//                   PINF("key:%lx value@%p value_len=%lu", key.c_str(), value, value_len);
//                     return 0;;
//                   });
// }

TEST_F(KVStore_test, BasicErase)
{
  _kvstore->erase(pool, "MyKey");
}

// TEST_F(KVStore_test, Allocate)
// {
//   uint64_t key_hash = 0;
//   ASSERT_TRUE(_kvstore->allocate(pool, "Elephant", MB(8), key_hash) == S_OK);
//   PLOG("Allocate: key_hash=%lx", key_hash);

//   ASSERT_TRUE(_kvstore->apply(pool, key_hash,
//                               [](void*p, const size_t plen) { memset(p,0xE,plen); }) == S_OK);

//   ASSERT_TRUE(_kvstore->apply(pool, key_hash,
//                               [](void*p, const size_t plen) { memset(p,0xE,plen); },
//                               KB(4),
//                               MB(2)) == S_OK);

//   /* out of bounds */
//   ASSERT_FALSE(_kvstore->apply(pool, key_hash,
//                                [](void*p, const size_t plen) { memset(p,0xE,plen); },
//                                MB(128),
//                                MB(2)) == S_OK);

// }

TEST_F(KVStore_test, ClosePool)
{
  _kvstore->close_pool(pool);
}




} // namespace

int main(int argc, char **argv) {
  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
