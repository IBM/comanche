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
#include <string>
#include <list>
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/str_utils.h>
#include <common/logging.h>
#include <common/cpu.h>
#include <common/utils.h>
#include <common/dump_utils.h>

#include <component/base.h>

#include <api/components.h>
#include <api/block_itf.h>
#include <api/region_itf.h>
#include <api/pager_itf.h>
#include <api/pmem_itf.h>

#define USE_SPDK_NVME_DEVICE // use SPDK-NVME or POSIX-NVME

#define DO_INTEGRITY
#define DO_STRESS_MEMORY
#define DO_STORE_RELOAD

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Pmem_fixed_test : public ::testing::Test {

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
  static Component::IBlock_device *      _block;
  static Component::IPersistent_memory * _pmem;
};

Component::IBlock_device * Pmem_fixed_test::_block;
Component::IPersistent_memory * Pmem_fixed_test::_pmem;


TEST_F(Pmem_fixed_test, InstantiateBlockDevice)
{
#ifdef USE_SPDK_NVME_DEVICE
  
  Component::IBase * comp = Component::load_component("libcomanche-blknvme.so",
                                                      Component::block_nvme_factory);

  assert(comp);
  PLOG("Block_device factory loaded OK.");
  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());

  cpu_mask_t cpus;
  cpus.add_core(2);
  _block = fact->create("8b:00.0",&cpus);

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
  //  config_string += "/dev/nvme0n1";
  config_string += "./blockfile.dat";
  //  config_string += "\"}";
  config_string += "\",\"size_in_blocks\":10000}";
  PLOG("config: %s", config_string.c_str());

  _block = fact->create(config_string);
  assert(_block);
  fact->release_ref();
  PINF("Block-layer component loaded OK (itf=%p)", _block);

#endif
}


TEST_F(Pmem_fixed_test, InstantiatePmem)
{
  using namespace Component;
  
  IBase * comp = load_component("libcomanche-pmemfixed.so",
                                Component::pmem_fixed_factory);
  assert(comp);
  IPersistent_memory_factory * fact = static_cast<IPersistent_memory_factory *>
    (comp->query_interface(IPersistent_memory_factory::iid()));
  assert(fact);
  _pmem = fact->open_allocator("testowner",_block);
  assert(_pmem);
  fact->release_ref();

  _pmem->start();
  
  PINF("Pmem-fixed component loaded OK.");
}

struct data_t {
  uint64_t val[512];
};


#ifdef DO_STORE_RELOAD
TEST_F(Pmem_fixed_test, StoreReload)
{
  size_t msize = KB(4) * 64; //MB(2) * 1;
  size_t n_elements = msize / sizeof(uint64_t);

  uint64_t * p = nullptr;
  size_t slab_size = n_elements * sizeof(uint64_t);

  IPersistent_memory::pmem_t h;
  bool reused;
  
  {/* set all zero */
    void* vptr = nullptr;
    h = _pmem->open("uint64array", slab_size, NUMA_NODE_ANY, reused, vptr);
    uint64_t * p = (uint64_t *) vptr;
    
    for(unsigned i=0;i<n_elements;i++) {
      p[i]=0;
    }
    _pmem->close(h);
  }



  {/* check all zero */
    void* vptr = nullptr;
    h = _pmem->open("uint64array", slab_size, NUMA_NODE_ANY, reused, vptr);
    uint64_t * p = (uint64_t *) vptr;
    
    for(unsigned i=0;i<n_elements;i++) {
      ASSERT_TRUE(p[i]==0);
    }
    _pmem->close(h);
    PLOG("Check zero OK.");
  }

  {/* write numbers in */
    void* vptr = nullptr;
    h = _pmem->open("uint64array", slab_size, NUMA_NODE_ANY, reused, vptr);
    uint64_t * p = (uint64_t *) vptr;

    for(uint64_t i=0;i<n_elements;i++) {
      p[i] = i;
    }
    _pmem->close(h); // implicit flush
    PLOG("written numbers");
  }

  {/* check numbers */
    void* vptr = nullptr;
    h = _pmem->open("uint64array", slab_size, NUMA_NODE_ANY, reused, vptr);
    uint64_t * p = (uint64_t *) vptr;

    for(uint64_t i=0;i<n_elements;i++) {
      if(p[i]!=i) {
        hexdump(&p[i],128);
      }
      ASSERT_TRUE(p[i] == i);
    }
    _pmem->close(h);
    PLOG("Check numbers OK.");
  }
  
#ifdef USE_SPDK_NVME_DEVICE 
  {/* partial flush */
    void* vptr = nullptr;
    h = _pmem->open("uint64array", slab_size, NUMA_NODE_ANY, reused, vptr);
    uint64_t * p = (uint64_t *) vptr;

    p[512] = 999;
        
    p[512*2] = 666;
    _pmem->persist_scoped(h, &p[512*2],8);

    _pmem->close(h, IPersistent_memory::FLAGS_NOFLUSH);
  }

  {/* partial flush */
    void* vptr = nullptr;
    h = _pmem->open("uint64array", slab_size, NUMA_NODE_ANY, reused, vptr);
    uint64_t * p = (uint64_t *) vptr;
    ASSERT_FALSE(p[512] == 999); // it didn't get flushed 
    ASSERT_TRUE(p[512*2] == 666);
    _pmem->close(h);
  }
#endif
  
}
#endif


#ifdef DO_INTEGRITY
TEST_F(Pmem_fixed_test, IntegrityCheck)
{
  size_t msize = KB(4) * 64; //MB(2) * 1;
  size_t n_elements = msize / sizeof(uint64_t);
  unsigned long ITERATIONS = 1000000UL;
  uint64_t * p = nullptr;
  size_t slab_size = n_elements * sizeof(uint64_t);
  bool reused;
  
  IPersistent_memory::pmem_t h = _pmem->open("integrityCheckBlock",slab_size,NUMA_NODE_ANY, reused, (void*&) p);

  PLOG("p=%p",p);
  ASSERT_FALSE(p==nullptr);


  /* 0xf check */
  for(unsigned long e=0;e<n_elements;e++)
    p[e] = 0xf;

  PINF("0xf writes complete. Starting check...");
  
  for(unsigned long e=0;e<n_elements;e++) {
    if(p[e] != 0xf) {
      PERR("Bad 0xf - check failed!");
      ASSERT_TRUE(0);
    }
  }
  PMAJOR("> 0xf check OK!");

  /* zero */
  //  memset(p,0,slab_size);
  for(unsigned long e=0;e<n_elements;e++)
    p[e] = 0x0;

  PINF("Zeroing complete.");

  for(unsigned long e=0;e<n_elements;e++) {
    if(p[e] != 0) {
      PERR("Bad zero not written through.");
      ASSERT_TRUE(0);
    }
  }
  PMAJOR("> Zero verification OK!");

  #if 1
  //  srand(rdtsc());
  for(unsigned long i=0;i<ITERATIONS;i++) {
    uint64_t slot = rand() % n_elements;
    if(p[slot] != 0 && p[slot] != slot) {
      PERR("Integrity check failed! slot-val=%ld expected=%ld or 0", p[slot], slot);
      ASSERT_TRUE(0);
    }    
    p[slot] = slot;

    if(i % 100000 == 0) {
      PLOG("Integrity iteration: %lu", i);
    }
  }
  #endif
    
  _pmem->close(h);
  PMAJOR("> Integrity check OK.");
}
#endif // DO_INTEGRITY



TEST_F(Pmem_fixed_test, ReleaseBlockDevice)
{
  ASSERT_TRUE(_pmem);
  ASSERT_TRUE(_block);

  _pmem->stop();
  _pmem->release_ref();
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();  
}
