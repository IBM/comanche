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
#include "data.h"
#include <gtest/gtest.h>
#include <common/utils.h>
#include <common/str_utils.h>
#include <core/physical_memory.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <api/block_itf.h>
#include <api/block_allocator_itf.h>

#include <stdlib.h>

#include <gperftools/profiler.h>


#define PMEM_PATH "/mnt/pmem0/pool-nvmestore"
#define POOL_NAME "test-basic.pool"


using namespace Component;


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
  static Component::IKVStore * _kvstore;
  static Component::IKVStore * _kvstore2;
  static Component::IKVStore::pool_t _pool;

};


Component::IKVStore * KVStore_test::_kvstore;
Component::IKVStore * KVStore_test::_kvstore2;
Component::IKVStore::pool_t KVStore_test::_pool;


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
  // pass blk and alloc here
  _pool = _kvstore->create_pool(PMEM_PATH, POOL_NAME, MB(128));
  ASSERT_TRUE(_pool > 0);
}

TEST_F(KVStore_test, BasicPut)
{
  ASSERT_TRUE(_pool);
  std::string key = "MyKey";
  std::string value = "Hello world!";
  //  value.resize(value.length()+1); /* append /0 */
  value.resize(MB(8));
    
  EXPECT_TRUE(S_OK == _kvstore->put(_pool, key, value.c_str(), value.length()));
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

  _kvstore->get_direct(_pool, key, value, value_len, 0);

  EXPECT_FALSE(strcmp("Hello world!", (char*)value));
  PINF("Value=(%.50s) %lu", ((char*)value), value_len);

  mem_alloc.free_io_buffer(handle);
}


TEST_F(KVStore_test, BasicGet)
{
  std::string key = "MyKey";

  void * value = nullptr;
  size_t value_len = 0;
  _kvstore->get(_pool, key, value, value_len);

  EXPECT_FALSE(strcmp("Hello world!", (char*)value));
  PINF("Value=(%.50s) %lu", ((char*)value), value_len);
}

// TEST_F(KVStore_test, BasicGetRef)
// {
//   std::string key = "MyKey";

//   void * value = nullptr;
//   size_t value_len = 0;
//   _kvstore->get_reference(_pool, key, value, value_len);
//   PINF("Ref Value=(%.50s) %lu", ((char*)value), value_len);
//   _kvstore->release_reference(_pool, value);
// }

#if 0

TEST_F(KVStore_test, BasicMap)
{
  _kvstore->map(_pool,[](uint64_t key,
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
  _kvstore->erase(_pool, "MyKey");
}

TEST_F(KVStore_test, Throughput)
{
  ASSERT_TRUE(_pool);

  size_t i;
  std::chrono::system_clock::time_point _start, _end;
  double secs;

  Data * _data = new Data();
  

  /* put */
  _start = std::chrono::high_resolution_clock::now();

  ProfilerStart("testnvmestore.put.profile");
  for(i = 0; i < _data->num_elements(); i ++){
    _kvstore->put(_pool, _data->key(i), _data->value(i), _data->value_len());
  }
  ProfilerStop();
  _end = std::chrono::high_resolution_clock::now();

  secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / 1000.0;
  PINF("*Put summary(with memcpy)*:");
  PINF("IOPS\tTP(MiB/s)\tValuesz(KiB)\tTime(s)\tnr_io");
  PINF("%2g\t%.2f\t%lu\t%.2lf\t%lu",
      ((double)i) / secs, ((double)i) / secs*_data->value_len()/(1024*1024), _data->value_len()/1024, secs, i);

  /* get */
  void * pval;
  size_t pval_len;

  _start = std::chrono::high_resolution_clock::now();
  ProfilerStart("testnvmestore.get.profile");
  for(i = 0; i < _data->num_elements(); i ++){
    _kvstore->get(_pool, _data->key(i), pval, pval_len);
  }
  ProfilerStop();
  _end = std::chrono::high_resolution_clock::now();

  secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / 1000.0;

  PINF("*Get summary*:");
  PINF("IOPS\tTP(MiB/s)\tValuesz(KiB)\tTime(s)\tnr_io");
  PINF("%2g\t%.2f\t%lu\t%.2lf\t%lu",
      ((double)i) / secs, ((double)i) / secs*_data->value_len()/(1024*1024), _data->value_len()/1024, secs, i);

  /* get direct */
  io_buffer_t handle;
  Core::Physical_memory  mem_alloc; // aligned and pinned mem allocator, TODO: should be provided through IZerocpy Memory interface of NVMestore

  handle = mem_alloc.allocate_io_buffer(_data->value_len(), 4096, Component::NUMA_NODE_ANY);
  ASSERT_TRUE(handle);
  pval = mem_alloc.virt_addr(handle);

  _start = std::chrono::high_resolution_clock::now();
  ProfilerStart("testnvmestore.get_direct.profile");
  for(i = 0; i < _data->num_elements(); i ++){
    _kvstore->get_direct(_pool, _data->key(i), pval, pval_len);
  }
  ProfilerStop();
  _end = std::chrono::high_resolution_clock::now();

  secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / 1000.0;

  PINF("*Get direct summary*:");
  PINF("IOPS\tTP(MiB/s)\tValuesz(KiB)\tTime(s)\tnr_io");
  PINF("%2g\t%.2f\t%lu\t%.2lf\t%lu",
      ((double)i) / secs, ((double)i) / secs*_data->value_len()/(1024*1024), _data->value_len()/1024, secs, i);

  mem_alloc.free_io_buffer(handle);
}

#if 0

TEST_F(KVStore_test, Allocate)
{
  uint64_t key_hash = 0;
  ASSERT_TRUE(_kvstore->allocate(_pool, "Elephant", MB(8), key_hash) == S_OK);
  PLOG("Allocate: key_hash=%lx", key_hash);

  PLOG("test 1");
  ASSERT_TRUE(_kvstore->apply(_pool, key_hash,
                              [](void*p, const size_t plen) { memset(p,0xE,plen); }) == S_OK);

  PLOG("test 2");
  ASSERT_TRUE(_kvstore->apply(_pool, key_hash,
                              [](void*p, const size_t plen) { memset(p,0xE,plen); },
                              KB(4),
                              MB(2)) == S_OK);

  /* out of bounds */
  PLOG("test 3");
  ASSERT_FALSE(_kvstore->apply(_pool, key_hash,
                               [](void*p, const size_t plen) { memset(p,0xE,plen); },
                               MB(128),
                               MB(2)) == S_OK);

}
#endif

#ifdef DO_ERASE
TEST_F(KVStore_test, ErasePool)
{
  _kvstore->delete_pool(_pool);
}
#endif

TEST_F(KVStore_test, ClosePool)
{
  _kvstore->close_pool(_pool);
}

/*
 * multiple store on same nvmedevice will use the same _block and the _blk_alloc
 */
TEST_F(KVStore_test, Multiplestore)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-nvmestore.so",
                                                      Component::nvmestore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  // this nvme-store use a block device and a block allocator
  _kvstore2 = fact->create("owner","name2", opt.pci.c_str());
  
  fact->release_ref();

  _pool = _kvstore2->create_pool(PMEM_PATH, "test-nvme2.pool", MB(128));
  _kvstore2->release_ref();
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
