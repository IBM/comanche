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


TEST_F(KVStore_test, Instantiate)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-storemap.so",
                                                      Component::mapstore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  _kvstore = fact->create("owner","name");
  
  fact->release_ref();
}

TEST_F(KVStore_test, OpenPool)
{
  ASSERT_TRUE(_kvstore);
  pool = _kvstore->create_pool("test1.pool", MB(32));

  if(pool == Component::IKVStore::POOL_ERROR)
    pool = _kvstore->open_pool("test1.pool");

  ASSERT_TRUE(pool != Component::IKVStore::POOL_ERROR);
}


TEST_F(KVStore_test, BasicPut)
{
  ASSERT_TRUE(pool);
  std::string key = "MyKey";
  std::string key2 = "MyKey2";
  std::string value = "Hello world!";
  //  value.resize(value.length()+1); /* append /0 */
  value.resize(MB(8));
    
  _kvstore->put(pool, key, value.c_str(), value.length());
  _kvstore->put(pool, key2, value.c_str(), value.length());
}

TEST_F(KVStore_test, BasicGet)
{
  std::string key = "MyKey";

  void * value = nullptr;
  size_t value_len = 0;
  _kvstore->get(pool, key, value, value_len);
  PINF("Value=(%.50s) %lu", ((char*)value), value_len);

  ASSERT_TRUE(value);
  ASSERT_TRUE(value_len == MB(8));

  value = nullptr;
  value_len = 0;
  _kvstore->get(pool, key, value, value_len);
  PINF("Repeat Value=(%.50s) %lu", ((char*)value), value_len);
  auto count = _kvstore->count(pool);
  PINF("Count = %ld", count);
  ASSERT_TRUE(count == 2);
  ASSERT_TRUE(value);
  ASSERT_TRUE(value_len == MB(8));
}


TEST_F(KVStore_test, BasicRemove)
{
  _kvstore->erase(pool, "MyKey");
}


TEST_F(KVStore_test, ClosePool)
{
  _kvstore->close_pool(pool);
}

TEST_F(KVStore_test, ReopenPool)
{
  pool = _kvstore->open_pool("test1.pool");
  ASSERT_TRUE(pool != Component::IKVStore::POOL_ERROR);
  PLOG("re-opened pool: %p", (void*) pool);
}

TEST_F(KVStore_test, ReClosePool)
{
  _kvstore->close_pool(pool);
}
  
TEST_F(KVStore_test, DeletePool)
{
  PLOG("deleting pool: %p", (void*) pool);
  _kvstore->delete_pool("test1.pool");
}



} // namespace

int main(int argc, char **argv) {
  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
