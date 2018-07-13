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

struct
{
  std::string pci;
} opt;

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
  static Component::IKVStore * _kvstore2;

  static std::string POOL_NAME;
};


Component::IKVStore * KVStore_test::_kvstore;
//Component::IKVStore * KVStore_test::_kvstore2;

std::string KVStore_test::POOL_NAME = "test-nvme";
constexpr static int nr_elem = 500; // number of the test elem in the pool

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
  _kvstore = fact->create("owner","name", opt.pci.c_str());
  
  fact->release_ref();
}

TEST_F(KVStore_test, OpenPool)
{
  PLOG(" test-nvmestore: try to openpool");
  ASSERT_TRUE(_kvstore);
  std::string pool_name = POOL_NAME + ".pool";
  // pass blk and alloc here
  pool = _kvstore->create_pool(PMEM_PATH, pool_name.c_str(), GB(8));
  ASSERT_TRUE(pool > 0);
}

TEST_F(KVStore_test, BasicPut)
{
  ASSERT_TRUE(pool);
  for(int i=0 ;i< nr_elem; i++){
    std::string key = "MyKey"+std::to_string(i);
    std::string value = "Hello world!"+std::to_string(i);
    //  value.resize(value.length()+1); /* append /0 */
    value.resize(MB(8));
      
    EXPECT_TRUE(S_OK == _kvstore->put(pool, key, value.c_str(), value.length()));
  }
}

#if 0
TEST_F(KVStore_test, GetDirect)
{

  /*Register Mem is only from gdr memory*/
  //ASSERT_TRUE(S_OK == _kvstore->register_direct_memory(user_buf, MB(8)));
  io_buffer_t handle;
  Core::Physical_memory  mem_alloc; // aligned and pinned mem allocator, TODO: should be provided through IZerocpy Memory interface of NVMestore
  std::string key = "MyKey0";
  void * value = nullptr;
  size_t value_len = 0;

  handle = mem_alloc.allocate_io_buffer(MB(8), 4096, Component::NUMA_NODE_ANY);
  ASSERT_TRUE(handle);
  value = mem_alloc.virt_addr(handle);

  _kvstore->get_direct(pool, key, value, value_len, 0);

  EXPECT_FALSE(strcmp("Hello world!0", (char*)value));
  PINF("Value=(%.50s) %lu", ((char*)value), value_len);

  mem_alloc.free_io_buffer(handle);
}
#endif

TEST_F(KVStore_test, BasicGet)
{

  for(int i=0 ;i< nr_elem; i++){
    std::string key = "MyKey"+ std::to_string(i);

    void * value = nullptr;
    size_t value_len = 0;
    _kvstore->get(pool, key, value, value_len);

    std::string expected_value = "Hello world!" + std::to_string(i);
    EXPECT_FALSE(strcmp(expected_value.c_str(), (char*)value));
    PINF("Value=(%.50s) %lu", ((char*)value), value_len);
  }
}

TEST_F(KVStore_test, BasicErase)
{

  for(int i=0 ;i< nr_elem; i++){
    _kvstore->erase(pool, "MyKey"+std::to_string(i));
  }

}


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
  if(argc!=2) {
    PINF("test-nvmestore <pci-address>");
    return 0;
  }

  opt.pci = argv[1];

  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
