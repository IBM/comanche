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
#include <mcheck.h>
#include <gtest/gtest.h>
#include <string>
#include <list>
#include <set>
#include <omp.h>
#include <chrono>
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/str_utils.h>
#include <common/logging.h>
#include <common/cpu.h>
#include <common/utils.h>
#include <common/dump_utils.h>
#include <common/str_utils.h>
#include <core/avl_malloc.h>
#include <core/dpdk.h>
#include <component/base.h>

#include <api/components.h>
#include <api/block_itf.h>
#include <api/region_itf.h>
#include <api/pmem_itf.h>
#include <api/block_allocator_itf.h>
#include <api/log_itf.h>

#define USE_SPDK_NVME_DEVICE // use SPDK-NVME or POSIX-NVME
#define IO_QUEUE_CORE_BASE 12

std::string device_name_arg;

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Log_store_test : public ::testing::Test {

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
  static Component::ILog   *        _log;
};

Component::IBlock_device * Log_store_test::_block;
Component::ILog * Log_store_test::_log;

TEST_F(Log_store_test, InitDPDK)
{
  DPDK::eal_init(2048);
}

TEST_F(Log_store_test, InstantiateBlockDevice)
{
#ifdef USE_SPDK_NVME_DEVICE
  
  Component::IBase * comp = Component::load_component("libcomanche-blknvme.so",
                                                      Component::block_nvme_factory);

  assert(comp);
  PLOG("Block_device factory loaded OK.");
  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  
  cpu_mask_t cpus;
  cpus.add_core(IO_QUEUE_CORE_BASE);
  //  cpus.add_core(13);

  _block = fact->create(device_name_arg.c_str(), &cpus);

  assert(_block);
  fact->release_ref();
  PINF("Lower block-layer component loaded OK.");

#else
  
  Component::IBase * comp = Component::load_component("libcomanche-blkposix.so",
                                                      Component::block_posix_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  std::string config_string;
  config_string = "{\"path\":\"";
  //  config_string += "/dev/nvme0n1";1
  config_string += "./blockfile.dat";
  //  config_string += "\"}";
  config_string += "\",\"size_in_blocks\":100000}";
  PLOG("config: %s", config_string.c_str());

  _block = fact->create(config_string);
  assert(_block);
  fact->release_ref();
  PINF("Block-layer component loaded OK (itf=%p)", _block);

#endif
}

#define RECORD_LEN 48

TEST_F(Log_store_test, Instantiate)
{
  Component::IBase * comp = Component::load_component("libcomanche-storelog.so",
                                                      Component::store_log_factory);
  assert(comp);
  PLOG("Log-store factory loaded OK.");

  ILog_factory * fact = (ILog_factory *) comp->query_interface(ILog_factory::iid());

  ASSERT_TRUE(_block);
  _log = fact->create("dwaddington",
                      "testlog",
                      _block,
                      FLAGS_FORMAT, //0,
                      RECORD_LEN,
                      false /* crc */);
  ASSERT_TRUE(_log);
  
  fact->release_ref();
}

#if 0
TEST_F(Log_store_test, CreateEntries)
{
  void* p = malloc(RECORD_LEN);

  size_t len = RECORD_LEN;

  auto started = std::chrono::high_resolution_clock::now();

  index_t idx;
  index_t last_index = _log->get_tail() - RECORD_LEN;
  PLOG("last index: %lu", last_index);

  unsigned long items = 0;
  for(unsigned i=0;i<GB(1)/len;i++) {
    //    _log->write(p, len, (omp_get_thread_num() % 2)+12);
    sprintf((char*)p,"Hello-%u",i);
    idx = _log->write(p, len); //, IO_QUEUE_CORE_BASE);//(i % 2) + 12);

    if(idx != (last_index + RECORD_LEN))
      throw General_exception("CreateEntries test failed; bad index (%ld expect %ld)",
                              idx, last_index+RECORD_LEN);
    last_index = idx;
    items++;
  }
  auto done = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();
  auto secs = ((float)ms)/1000.0f;
  PINF("Duration %f seconds", secs);
  PINF("Rate: %f M items per second", (items / secs)/1000000.0);
  _log->flush();
  free(p);
}
#endif

TEST_F(Log_store_test, ReadEntries)
{
  auto iob = _log->allocate_io_buffer(MB(4),KB(4),NUMA_NODE_ANY);
  void * iob_p = _log->virt_addr(iob);
  memset(iob_p,0,RECORD_LEN+1);
  
  for(unsigned i=0;i<10;i++) {
    void * r = _log->read(i, iob);
    PLOG("[%u]: %s", i, (char*)r);
  }
  _log->free_io_buffer(iob);
}
    
// TEST_F(Log_store_test, CreateEntries)
// {
//   size_t total_write_size = GB(2);
//   std::string val = Common::random_string(MB(1));

//   // std::vector<std::string> keys;
//   // for(unsigned i=0;i<total_write_size/MB(1);i++) {
//   //   keys.push_back(Common::random_string(128));
//   // }

//   auto started = std::chrono::high_resolution_clock::now();

//   // //#pragma omp parallel for shared(keys)
//   // for(unsigned i=0;i<total_write_size/MB(1);i++) {
//   //   _log->put(keys[i], "metadata", (void*)val.c_str(), val.size());
//   // }
//   // _log->flush();
  
//   auto done = std::chrono::high_resolution_clock::now();
//   auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();
//   PLOG("Duration %f seconds: %lu transactions", ((float)ms)/1000.0, total_write_size/MB(1));
// }

TEST_F(Log_store_test, ReleaseBlockDevice)
{
  ASSERT_TRUE(_log);
  ASSERT_TRUE(_block);

  _log->release_ref();
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  mtrace();

  if(argc > 1)
    device_name_arg = argv[1];
  else
    device_name_arg = "0b:00.0";
  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  muntrace();
  return r;
}
