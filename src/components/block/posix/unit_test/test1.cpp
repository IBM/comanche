/*
   Copyright [2017] [IBM Corporation]

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
#include <string>
#include <vector>
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/cpu.h>

#include <component/base.h>
#include <api/components.h>
#include <api/block_itf.h>
#include <api/fs_itf.h>

using namespace Component;

struct
{
  std::string filename;
} opt;

namespace {

// The fixture for testing class Foo.
class Block_posix_test : public ::testing::Test {

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
};


Component::IBlock_device * Block_posix_test::_block;

TEST_F(Block_posix_test, InstantiateBlockDevice)
{
  Component::IBase * comp = Component::load_component("libcomanche-blkposix.so",
                                                      Component::block_posix_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  std::string config_string;
  if(opt.filename.substr(0,4)=="/dev") {
    config_string = "{\"path\":\"";
    config_string += opt.filename;
    config_string += "\"}";
  }
  else {
    config_string = "{\"path\":\"";
    config_string += opt.filename;
    config_string += "\",\"size_in_blocks\" : 4000 }";
  }
  PLOG("config: %s", config_string.c_str());
  
  _block = fact->create(config_string);
  assert(_block);
  fact->release_ref();
  PINF("POSIX-based block-layer component loaded OK.");
}

TEST_F(Block_posix_test, MemoryAllocation)
{
  std::vector<io_buffer_t> handles;
  for(unsigned i=0;i<20;i++) {
    size_t size = rand() % MB(4);    
    io_buffer_t mem = _block->allocate_io_buffer(size,4096,Component::NUMA_NODE_ANY);
    void * ptr = _block->virt_addr(mem);
    PLOG("ptr=%p", ptr);
    ASSERT_TRUE(check_aligned(ptr, 4096));
    ASSERT_TRUE(check_aligned(((void*)_block->phys_addr(mem)), 4096));
    memset(ptr, 0, size);
    handles.push_back(mem);
  }
  for(auto& h: handles) {
    _block->free_io_buffer(h);
  }
}

TEST_F(Block_posix_test, BasicAsync)
{
  io_buffer_t mem = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  void * ptr = _block->virt_addr(mem);
  char *p = (char*) ptr;
  
  unsigned BLOCK_COUNT = 10;
  for(unsigned i=0;i<4096;i++)
    p[i] = i % 0xff;

  
#if 0
  _block->write(mem, 0, 1, 1);
#else
  uint64_t tag1 = _block->async_write(mem, 0, 1, 1);
  while(!_block->check_completion(tag1)) { usleep(1000); }
#endif
  
  memset(ptr, 0x0, 4096);

#if 0
  _block->read(mem, 0, 1, 1);
#else
  uint64_t tag2 = _block->async_read(mem, 0, 1, 1);
  while(!_block->check_completion(tag2)) { usleep(1000); }
#endif
  
  for(unsigned i=0;i<4096;i++) {
    if((p[i] & 0xff) != (i % 0xff)) {
      PWRN("eeek !! p[%u]=%x",i,p[i]);
      ASSERT_TRUE(0);
    }
  }
  
  _block->free_io_buffer(mem);
  PMAJOR("> basic async test OK");
}

#if 0
TEST_F(Block_posix_test, MultiAsyncAggCheck)
{
  unsigned NUM_PAGES = 128;
  io_buffer_t mem = _block->allocate_io_buffer(NUM_PAGES * PAGE_SIZE, PAGE_SIZE, Component::NUMA_NODE_ANY);
  byte * p = (byte*) _block->virt_addr(mem);
  PINF("IO buffer: %p", p);

  memset(p,0,NUM_PAGES * PAGE_SIZE);

  for(unsigned n=0;n<NUM_PAGES;n++) {
    ((int*)p)[n] = n;
  }

  uint64_t gwid;
  for(unsigned n=0;n<NUM_PAGES;n++) {
    gwid = _block->async_write(mem, n*PAGE_SIZE, n, 1);
  }
  while(!_block->check_completion(gwid));

  memset(p,0,NUM_PAGES * PAGE_SIZE);
  for(unsigned n=0;n<NUM_PAGES;n++) {
    gwid = _block->async_read(mem, n*PAGE_SIZE, n, 1);
  }
  while(!_block->check_completion(gwid));

  for(unsigned n=0;n<NUM_PAGES;n++) {
    ASSERT_TRUE(((int*)p)[n] == n);
  }

  PMAJOR("> MultiAsyncAggCheck complete.");
}

#endif

#if 0
TEST_F(Block_posix_test, PartitionIntegrity)
{
  using namespace Component;
  
  io_buffer_t mem = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  void * ptr = _block->virt_addr(mem);
  unsigned ITERATIONS = 10000;

  VOLUME_INFO vinfo;
  _block->get_volume_info(vinfo);
  
  PLOG("Volume Info: size=%ld blocks", vinfo.max_lba);
  PLOG("             block_size=%u", vinfo.block_size);
  PLOG("             name=%s", vinfo.volume_name);

  unsigned BLOCK_COUNT = vinfo.max_lba;

  /* zero blocks first */
  memset(ptr,0,4096);

  uint64_t tag;
  for(unsigned k=0;k < BLOCK_COUNT; k++) {
    tag = _block->async_write(mem, 0, k, 1);
    while(!_block->check_completion(tag)) usleep(100);
  }
    
  PMAJOR("> Zeroing complete OK.");

  for(unsigned i=0;i<ITERATIONS;i++) {
    uint64_t lba = rand() % BLOCK_COUNT;
    
    /* read existing content */
    tag = _block->async_read(mem, 0, lba, 1);
    while(!_block->check_completion(tag)) usleep(1000);
    
    uint64_t * v = (uint64_t*) ptr;
    if(v[0] != 0 && v[0] != lba) {
      PERR("value read from drive = %lx, lba=%ld", *v,lba);
      throw General_exception("bad data!");
    }
    /* TODO: check rest of block? */

    /* write out LBA into first 64bit */
    v[0] = lba;
    tag = _block->async_write(mem, 0, lba, 1);
    while(!_block->check_completion(tag)) usleep(1000);
  }

  _block->free_io_buffer(mem);

  PINF("Integrity check OK.");
}
#endif


#if 1
TEST_F(Block_posix_test, PartitionIntegritySync)
{
  using namespace Component;
  
  io_buffer_t mem = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  void * ptr = _block->virt_addr(mem);
  unsigned ITERATIONS = 100;

  VOLUME_INFO vinfo;
  _block->get_volume_info(vinfo);
  
  PLOG("Volume Info: size=%ld blocks", vinfo.max_lba);
  PLOG("             block_size=%u", vinfo.block_size);
  PLOG("             name=%s", vinfo.volume_name);

  unsigned BLOCK_COUNT = vinfo.max_lba;

  /* zero blocks first */
  memset(ptr,0,4096);

  uint64_t tag;
  for(unsigned k=0;k < BLOCK_COUNT; k++) {
    _block->write(mem, 0, k, 1);
  }
    
  PMAJOR("> Zeroing complete OK.");

  for(unsigned i=0;i<ITERATIONS;i++) {
    uint64_t lba = rand() % BLOCK_COUNT;
    
    /* read existing content */
    _block->read(mem, 0, lba, 1);
    
    uint64_t * v = (uint64_t*) ptr;
    if(v[0] != 0 && v[0] != lba) {
      PERR("value read from drive = %lx, lba=%ld", *v,lba);
      throw General_exception("bad data!");
    }
    /* TODO: check rest of block? */

    /* write out LBA into first 64bit */
    v[0] = lba;
    _block->write(mem, 0, lba, 1);
  }

  _block->free_io_buffer(mem);

  PINF("Integrity check OK.");
}
#endif

TEST_F(Block_posix_test, GetPhysAddr)
{
  io_buffer_t mem = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  void * ptr = _block->virt_addr(mem);
  addr_t phys = _block->phys_addr(mem);
  PLOG("phys lookup:%lx", phys);
  ASSERT_TRUE(phys > 0);
  _block->free_io_buffer(mem);
}

TEST_F(Block_posix_test, AsyncWriteWithCallback)
{
  io_buffer_t buffer = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  IBlock_device::Semaphore sem;
  
  _block->async_write(buffer,
                      0,
                      0,
                      1, // lba_count
                      0, // queue_id
                      [](uint64_t gwid, void* arg0, void* arg1)
                      {
                        PLOG("Callback: gwid=%lu arg0=%p arg1=%p",gwid, arg0, arg1);
                        ((IBlock_device::Semaphore *)arg0)->post();
                        ASSERT_TRUE(arg1 == (void*) 0xBEEF);
                      },
                      (void*) &sem,
                      (void*) 0xBEEF);

  sem.wait();
  PLOG("Whooo!");
  _block->free_io_buffer(buffer);
}


TEST_F(Block_posix_test, AsyncReadWithCallback)
{
  io_buffer_t buffer = _block->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  IBlock_device::Semaphore sem;
  
  _block->async_read(buffer,
                     0,
                     0,
                     1, // lba_count
                     0, // queue_id
                     [](uint64_t gwid, void* arg0, void* arg1)
                     {
                       PLOG("Callback: gwid=%lu arg0=%p arg1=%p",gwid, arg0, arg1);
                       ((IBlock_device::Semaphore *)arg0)->post();
                       ASSERT_TRUE(arg1 == (void*) 0xBEEF);
                     },
                     (void*) &sem,
                     (void*) 0xBEEF);

  sem.wait();
  PLOG("Whooo!");
  _block->free_io_buffer(buffer);
}
  

TEST_F(Block_posix_test, ReleaseBlockDevice)
{
  assert(_block);
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  if(argc!=2) {
    PINF("test <filename>");
    return 0;
  }

  opt.filename = argv[1];
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
