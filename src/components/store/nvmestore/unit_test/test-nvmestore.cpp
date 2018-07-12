/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Feng Li (fengggli@yahoo.com)
 *
 */


/* note: we do not include component source, only the API definition */
#include <gtest/gtest.h>
#include <common/utils.h>
#include <common/str_utils.h>
#include <core/physical_memory.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <api/block_itf.h>
#include <api/block_allocator_itf.h>
#include "data.h"

#include <stdlib.h>

#include <gperftools/profiler.h>



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
  static Component::IBlock_device * _block;
  static Component::IBlock_allocator * _alloc;
  static Component::IKVStore * _kvstore;

  static std::string POOL_NAME;
};


Component::IKVStore * KVStore_test::_kvstore;

std::string KVStore_test::POOL_NAME = "test-nvme";

#define PMEM_PATH "/mnt/pmem0/pool/0/"
//#define PMEM_PATH "/dev/pmem0"


TEST_F(KVStore_test, Instantiate)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-nvmestore.so",
                                                      Component::nvmestore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  // this nvme-store use a block device and a block allocator
  _kvstore = fact->create("owner","name");
  
  fact->release_ref();
}

TEST_F(KVStore_test, OpenPool)
{
  PLOG(" test-nvmestore: try to openpool");
  ASSERT_TRUE(_kvstore);
  std::string pool_name = POOL_NAME + ".pool";
  // pass blk and alloc here
  pool = _kvstore->create_pool(PMEM_PATH, pool_name.c_str(), MB(128));
  ASSERT_TRUE(pool > 0);
}

TEST_F(KVStore_test, BasicPut)
{
  ASSERT_TRUE(pool);
  std::string key = "MyKey";
  std::string value = "Hello world!";
  //  value.resize(value.length()+1); /* append /0 */
  value.resize(MB(8));
    
  EXPECT_TRUE(S_OK == _kvstore->put(pool, key, value.c_str(), value.length()));
}

TEST_F(KVStore_test, GetDirect)
{

  /*Register Mem is only from gdr memory*/
  //ASSERT_TRUE(S_OK == _kvstore->register_direct_memory(user_buf, MB(8)));
  io_buffer_t handle;
  Core::Physical_memory  mem_alloc; // aligned and pinned mem allocator, TODO: should be provided through IZerocpy Memory interface of NVMestore
  std::string key = "MyKey";
  void * value = nullptr;
  size_t value_len = 0;

  handle = mem_alloc.allocate_io_buffer(MB(8), 4096, Component::NUMA_NODE_ANY);
  ASSERT_TRUE(handle);
  value = mem_alloc.virt_addr(handle);

  _kvstore->get_direct(pool, key, value, value_len, 0);

  EXPECT_FALSE(strcmp("Hello world!", (char*)value));
  PINF("Value=(%.50s) %lu", ((char*)value), value_len);

  mem_alloc.free_io_buffer(handle);
}


TEST_F(KVStore_test, BasicGet)
{
  std::string key = "MyKey";

  void * value = nullptr;
  size_t value_len = 0;
  _kvstore->get(pool, key, value, value_len);

  EXPECT_FALSE(strcmp("Hello world!", (char*)value));
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

#if 0

TEST_F(KVStore_test, BasicMap)
{
  _kvstore->map(pool,[](uint64_t key,
                        const void * value,
                        const size_t value_len) -> int
                {
                    PINF("key:%lx value@%p value_len=%lu", key, value, value_len);
                    return 0;;
                  });
}
#endif

TEST_F(KVStore_test, BasicErase)
{
  _kvstore->erase(pool, "MyKey");
}

#if 0
TEST_F(KVStore_test, Throughput)
{
  ASSERT_TRUE(pool);

  size_t i;
  std::chrono::system_clock::time_point _start, _end;
  double secs;

  Data * _data = new Data();
  

  /* put */
  _start = std::chrono::high_resolution_clock::now();

  ProfilerStart("testnvmestore.put.profile");
  for(i = 0; i < _data->num_elements(); i ++){
    _kvstore->put(pool, _data->key(i), _data->value(i), _data->value_len());
  }
  ProfilerStop();
  _end = std::chrono::high_resolution_clock::now();

  secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / 1000.0;
  PINF("*Put* IOPS: %2g, %.2lf seconds for %lu operations",  ((double)i) / secs, secs, i);

  /* get */
  void * pval;
  size_t pval_len;

  _start = std::chrono::high_resolution_clock::now();
  ProfilerStart("testnvmestore.get.profile");
  for(i = 0; i < _data->num_elements(); i ++){
    _kvstore->get(pool, _data->key(i), pval, pval_len);
  }
  ProfilerStop();
  _end = std::chrono::high_resolution_clock::now();

  secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / 1000.0;
  PINF("*Get* IOPS: %2g, %.2lf seconds for %lu operations",  ((double)i) / secs, secs, i);


}
#endif

#if 0

TEST_F(KVStore_test, Allocate)
{
  uint64_t key_hash = 0;
  ASSERT_TRUE(_kvstore->allocate(pool, "Elephant", MB(8), key_hash) == S_OK);
  PLOG("Allocate: key_hash=%lx", key_hash);

  PLOG("test 1");
  ASSERT_TRUE(_kvstore->apply(pool, key_hash,
                              [](void*p, const size_t plen) { memset(p,0xE,plen); }) == S_OK);

  PLOG("test 2");
  ASSERT_TRUE(_kvstore->apply(pool, key_hash,
                              [](void*p, const size_t plen) { memset(p,0xE,plen); },
                              KB(4),
                              MB(2)) == S_OK);

  /* out of bounds */
  PLOG("test 3");
  ASSERT_FALSE(_kvstore->apply(pool, key_hash,
                               [](void*p, const size_t plen) { memset(p,0xE,plen); },
                               MB(128),
                               MB(2)) == S_OK);

}
#endif

#ifdef DO_ERASE
TEST_F(KVStore_test, ErasePool)
{
  _kvstore->delete_pool(pool);
}
#endif

TEST_F(KVStore_test, ClosePool)
{
  _kvstore->close_pool(pool);
}

TEST_F(KVStore_test, ReleaseStore)
{
  _kvstore->release_ref();
}




} // namespace

int main(int argc, char **argv) {
  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
