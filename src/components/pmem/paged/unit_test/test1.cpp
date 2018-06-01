#include <gtest/gtest.h>
#include <string>
#include <random>
#include <list>
#include <common/cycles.h>
#include <common/exceptions.h>
#include <common/str_utils.h>
#include <common/logging.h>
#include <common/cpu.h>
#include <common/utils.h>

#include <component/base.h>

#include <api/components.h>
#include <api/block_itf.h>
#include <api/region_itf.h>
#include <api/pager_itf.h>
#include <api/pmem_itf.h>

//#define USE_SPDK_NVME_DEVICE // use SPDK-NVME or POSIX-NVME

#define DO_INTEGRITY
#define DO_STRESS_MEMORY

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Pmem_paged_test : public ::testing::Test {

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
  static Component::IPager *             _pager;
  static Component::IPersistent_memory * _pmem;
};

Component::IBlock_device * Pmem_paged_test::_block;
Component::IPager * Pmem_paged_test::_pager;
Component::IPersistent_memory * Pmem_paged_test::_pmem;


TEST_F(Pmem_paged_test, InstantiateBlockDevice)
{
#ifdef USE_SPDK_NVME_DEVICE
  
  Component::IBase * comp = Component::load_component("libcomanche-blknvme.so",
                                                      Component::block_nvme_factory);

  assert(comp);
  PLOG("Block_device factory loaded OK.");
  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());

  cpu_mask_t mask;
  mask.add_core(2);
  _block = fact->create("81:00.0", &mask);

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



#define NUM_PAGER_PAGES 128

TEST_F(Pmem_paged_test, InstantiatePager)
{
  using namespace Component;
  
  assert(_block);

  IBase * comp = load_component("libcomanche-pagersimple.so",
                                Component::pager_simple_factory);
  assert(comp);
  IPager_factory * fact = static_cast<IPager_factory *>(comp->query_interface(IPager_factory::iid()));
  assert(fact);
  _pager = fact->create(NUM_PAGER_PAGES,"unit-test-heap",_block, true /* force init */);
  assert(_pager);

  PINF("Pager-simple component loaded OK.");
}

TEST_F(Pmem_paged_test, InstantiatePmem)
{
  using namespace Component;
  
  assert(_pager);

  IBase * comp = load_component("libcomanche-pmempaged.so",
                                Component::pmem_paged_factory);
  assert(comp);
  IPersistent_memory_factory * fact = static_cast<IPersistent_memory_factory *>
    (comp->query_interface(IPersistent_memory_factory::iid()));
  assert(fact);
  _pmem = fact->open_allocator("testowner",_pager);
  assert(_pmem);
  fact->release_ref();

  _pmem->start();
  
  PINF("Pmem-pager component loaded OK.");
}

struct data_t {
  uint64_t val[512];
};

#ifdef DO_INTEGRITY
TEST_F(Pmem_paged_test, IntegrityCheck)
{
  size_t msize = KB(4) * 64; // works OK. but only mapped once
  //size_t msize = KB(4) * 1024; // broken!
  size_t n_elements = msize / sizeof(uint64_t);
  unsigned long ITERATIONS = 1000000UL;
  uint64_t * p = nullptr;
  size_t slab_size = n_elements * sizeof(uint64_t);
  bool reused;

  IPersistent_memory::pmem_t handle = _pmem->open("integrityCheckBlock", slab_size, NUMA_NODE_ANY, reused, (void*&)p);

  PLOG("handle: %p", handle);
  ASSERT_FALSE(p==nullptr);


  /* 0xf check */
  for(unsigned long e=0;e<n_elements;e++)
    p[e] = 0xf;

  PINF("0xf writes complete. Starting check...");

  PLOG("current total number of faults: %lu", _pmem->fault_count());
  
  for(unsigned long e=0;e<n_elements;e++) {
    if(p[e] != 0xf) {
      PERR("Bad 0xf - check failed!");
      ASSERT_TRUE(0);
    }
  }
  PMAJOR("> 0xf check OK!");

  /* zero */
  memset(p,0,slab_size);
  PINF("Zeroing complete.");

  for(unsigned long e=0;e<n_elements;e++) {
    if(p[e] != 0) {
      PERR("Bad zero not written through.");
      ASSERT_TRUE(0);
    }
  }
  PMAJOR("> Zero verification OK!");

  /* integrity test */
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

  PLOG("Closing pmem handle");
  _pmem->close(handle);
  PMAJOR("> Integrity check OK.");
}
#endif // DO_INTEGRITY

#ifdef DO_STRESS_MEMORY
TEST_F(Pmem_paged_test, UseMemory)
{  
  size_t n_elements = 1000;
  char * p = nullptr;
  bool reused;

  PLOG("Performing pf/sec test...");
  auto handle = _pmem->open("someAllocId",n_elements*PAGE_SIZE, NUMA_NODE_ANY, reused, (void*&) p);
  
  PLOG("pmem->allocate gave %p", p);

  ASSERT_FALSE(p==nullptr);

  unsigned long count = 0;

#if 0 /* stride write */
  const unsigned HOP=8;
  PLOG("Doing stride write.. hop=%u", HOP);
  assert(n_elements % HOP == 0);

  cpu_time_t begin = rdtsc();
      
  for(unsigned long h=0;h<HOP;h++) {
    for(unsigned long i=h;i<n_elements;i+=HOP) {
      p[PAGE_SIZE*i] = 0xf;
      if(++count % 10000 == 0) PLOG("count: %lu", count);
    }
  }

  cpu_time_t end = rdtsc();
  uint64_t usec = (end - begin) / 2400;
  PINF("%lu usec/page fault", usec / n_elements);
#endif

#if 1 /* random write */
std::vector<addr_t> pages;

  PLOG("Creating random write sequence..");
  for(unsigned long i=0;i<n_elements;i++) {
    pages.push_back(i);
  }
  std::random_device rd;
  std::default_random_engine e(0);
  std::mt19937 g(e());
 
  std::shuffle(pages.begin(), pages.end(), g);

  PLOG("Starting random write..");
  size_t fcount_start = _pmem->fault_count();
  
  /* seq write */
  cpu_time_t begin = rdtsc();

  const unsigned ITERATIONS = 100;
  for(unsigned i=0;i<ITERATIONS;i++) {
    for(auto& i: pages) {
      p[PAGE_SIZE*i] = 0xf;
      if(++count % 100000 == 0) PLOG("count: %lu", count);
    }
  }

  cpu_time_t end = rdtsc();
  uint64_t usec = (end - begin) / 2400;
  size_t fcount = _pmem->fault_count() - fcount_start;
  PINF("%lu usec/page fault (over %ld faults)!",
       usec / fcount, fcount);

#endif
  
#if 0 /* seq write */  
  cpu_time_t begin = rdtsc();

  for(unsigned i=0;i<n_elements;i++) {
    p[PAGE_SIZE*i + 2] = 0xf;
    if(++count % 100000 == 0) PLOG("count: %lu", count);
  }
  cpu_time_t end = rdtsc();
  uint64_t usec = (end - begin) / 2400;
  PINF("%lu usec/page fault", usec / n_elements);
#endif

#if 0 /* rand write */
  const unsigned ITERATIONS = 1000;
  for(unsigned i=0;i<ITERATIONS;i++) {
    unsigned idx = rand() % (n_elements*PAGE_SIZE);
    p[idx]++;
  }
#endif
  
  _pmem->close(handle);
}
#endif


TEST_F(Pmem_paged_test, ReleaseBlockDevice)
{
  ASSERT_TRUE(_pmem);
  ASSERT_TRUE(_block);

  PLOG("total number of faults: %lu", _pmem->fault_count());

  _pmem->stop();
  _pmem->release_ref();
  _pager->release_ref();
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
