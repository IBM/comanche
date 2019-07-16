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


//#define USE_FILESTORE
#undef USE_FILESTORE


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
  static bool _pool_is_reopen;  // this run is open a previously created pool

  using kv_t = std::tuple<std::string, std::string>;
  static std::vector<kv_t> kvv;
};


Component::IKVStore * KVStore_test::_kvstore;
Component::IKVStore * KVStore_test::_kvstore2;
Component::IKVStore::pool_t KVStore_test::_pool;
bool KVStore_test::_pool_is_reopen;

std::vector<KVStore_test::kv_t> KVStore_test::kvv;

TEST_F(KVStore_test, Instantiate)
{
  /* create object instance through factory */
#ifndef USE_FILESTORE
  Component::IBase * comp = Component::load_component("libcomanche-nvmestore.so",
                                                      Component::nvmestore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  std::map<std::string, std::string> params;
  params["owner"] = "testowner";
  params["name"] = "testname";
  params["pci"] = opt.pci;
  params["pm_path"] = "/mnt/pmem0/";
  params["persist_type"] = "hstore";
  unsigned debug_level = 0;

  _kvstore = fact->create(debug_level, params);
#else
  Component::IBase * comp = Component::load_component("libcomanche-storefile.so",
                                                      Component::filestore_factory);

  ASSERT_TRUE(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  // this nvme-store use a block device and a block allocator
  _kvstore = fact->create("owner","name");
#endif

  fact->release_ref();
}

TEST_F(KVStore_test, OpenPool)
{
  PLOG(" test-nvmestore: try to openpool");
  ASSERT_TRUE(_kvstore);
  // pass blk and alloc here
  std::string pool_path;
  std::string pool_name;

  pool_name = "nvmestore-tp-nr-"+std::to_string(Data::NUM_ELEMENTS) + "-sz-" + std::to_string(Data::VAL_LEN)+ ".pool";

  pool_path = "./data/";
  try{
  _pool = _kvstore->create_pool(pool_path + pool_name, MB(128));
  _pool_is_reopen = false;
  }
  catch(...){
    // open the pool if it exists
    _pool = _kvstore->open_pool(pool_path + pool_name);
    _pool_is_reopen = true;
    PINF("NVMEStore:open a exsiting pool instead!");
  }
  ASSERT_TRUE(_pool > 0);
}


TEST_F(KVStore_test, ThroughputPut)
{
  ASSERT_TRUE(_pool);
  if(_pool_is_reopen){
    PINF("open a exisitng pool, skip PUT");
  }
  else{
    size_t i;
    std::chrono::system_clock::time_point _start, _end;
    double secs;

    Data *_data = new Data();

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
    }
}

TEST_F(KVStore_test, ThroughputGetDirect){

  void * pval;
  size_t pval_len;

  ASSERT_TRUE(_pool);

  size_t i;
  std::chrono::system_clock::time_point _start, _end;
  double secs;


#if 0
  /* get */

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
      ((double)i) / secs, ((double)i) / secs*Data::VAL_LEN/(1024*1024), Data::VAL_LEN/1024, secs, i);
#endif

  /* get direct */

#ifndef USE_FILESTORE
  IKVStore::memory_handle_t handle;
  ASSERT_EQ(S_OK, _kvstore->allocate_direct_memory(pval, Data::VAL_LEN, handle));
  
#else
  pval = malloc(Data::VAL_LEN);
  ASSERT_TRUE(pval);
#endif
  pval_len = Data::VAL_LEN;

  int ret;

  PINF("*IO memory allocated*:");
  _start = std::chrono::high_resolution_clock::now();
  ProfilerStart("testnvmestore.get_direct.profile");
  for(i = 0; i < Data::NUM_ELEMENTS; i ++){
    std::string key = "elem" + std::to_string(i);
    if(ret  = _kvstore->get_direct(_pool, key, pval, pval_len, handle)){
      throw API_exception("NVME_store:: get erorr with return value %d", ret);
    }
  }
  ProfilerStop();
  _end = std::chrono::high_resolution_clock::now();

  secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / 1000.0;

  PINF("*Get direct summary*:");
  PINF("IOPS\tTP(MiB/s)\tValuesz(KiB)\tTime(s)\tnr_io");
  PINF("%2g\t%.2f\t%lu\t%.2lf\t%lu",
      ((double)i) / secs, ((double)i) / secs*Data::VAL_LEN/(1024*1024), Data::VAL_LEN/1024, secs, i);

#ifndef USE_FILESTORE
  _kvstore->free_direct_memory(handle);
#else
  free(pval);
#endif
}


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
