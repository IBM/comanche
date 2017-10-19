#include <gtest/gtest.h>
#include <string>
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

#define USE_SPDK_NVME_DEVICE // use UNVME or SPDK-NVME

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
  static Component::IRegion_manager *    _rm;
  static Component::IPager *             _pager;
  static Component::IPersistent_memory * _pmem;
};

Component::IBlock_device * Pmem_paged_test::_block;
Component::IRegion_manager * Pmem_paged_test::_rm;
Component::IPager * Pmem_paged_test::_pager;
Component::IPersistent_memory * Pmem_paged_test::_pmem;

TEST_F(Pmem_paged_test, InstantiateBlockDevice)
{
#ifdef USE_SPDK_NVME_DEVICE
  
  std::string dll_path = getenv("COMANCHE_HOME");
  dll_path.append("/lib/libcomanche-blk.so");
  Component::IBase * comp = Component::load_component(dll_path.c_str(),
                                                      Component::block_nvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  _block = fact->create("./vol-config-local.json");
  assert(_block);
  fact->release_ref();
  PINF("Lower block-layer component loaded OK.");

#else
  
  std::string dll_path = getenv("COMANCHE_HOME");
  dll_path.append("/lib/libcomanche-blkunvme.so");
  Component::IBase * comp = Component::load_component(dll_path.c_str(),
                                                      Component::block_unvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());  
  _block = fact->create("8b:00.0");
  assert(_block);
  fact->release_ref();
  PINF("uNVME-based block-layer component loaded OK.");

#endif
}

TEST_F(Pmem_paged_test, InstantiateRm)
{
  using namespace Component;
  
  ASSERT_TRUE(_block);
  
  std::string dll_path = getenv("COMANCHE_HOME");
  dll_path += "/lib/libcomanche-partregion.so";

  IBase * comp = load_component(dll_path.c_str(),
                                Component::part_region_factory);
  assert(comp);
  IRegion_manager_factory* fact =
    (IRegion_manager_factory *) comp->query_interface(IRegion_manager_factory::iid());
  
  assert(fact);

  /* pass in lower-level block device */
  _rm = fact->open(_block, IRegion_manager_factory::FLAGS_FORMAT);
  
  ASSERT_TRUE(_rm);
  fact->release_ref();
  
  PINF("Part-region component loaded OK.");
}

TEST_F(Pmem_paged_test, InstantiatePager)
{
  using namespace Component;
  
  assert(_block);
  std::string dll_path = getenv("COMANCHE_HOME");
  dll_path += "/lib/libcomanche-pagersimple.so";

  IBase * comp = load_component(dll_path.c_str(),
                                Component::pager_simple_factory);
  assert(comp);
  IPager_factory * fact = static_cast<IPager_factory *>(comp->query_interface(IPager_factory::iid()));
  assert(fact);
  _pager = fact->create(32,"unit-test-heap",_block);
  assert(_pager);

  PINF("Pager-simple component loaded OK.");

}

TEST_F(Pmem_paged_test, InstantiatePmem)
{
  using namespace Component;
  
  assert(_pager);
  std::string dll_path = getenv("COMANCHE_HOME");
  dll_path += "/lib/libcomanche-pmempaged.so";

  IBase * comp = load_component(dll_path.c_str(),
                                Component::pmem_paged_factory);
  assert(comp);
  IPersistent_memory_factory * fact = static_cast<IPersistent_memory_factory *>
    (comp->query_interface(IPersistent_memory_factory::iid()));
  assert(fact);
  _pmem = fact->open_heap_set("UnitestHeapSet", _pager);
  assert(_pmem);
  fact->release_ref();

  _pmem->start();
  
  PINF("Pmem-pager component loaded OK.");
}

struct data_t {
  uint64_t val[512];
};

#if 1
TEST_F(Pmem_paged_test, UseMemory)
{
  size_t n_elements = 8000000; // 80M
  char * p = nullptr;

  PLOG("Performing pf/sec test...");
  IPersistent_memory::pmem_t pmem = _pmem->allocate("someAllocId",n_elements*PAGE_SIZE,(void**)&p);

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

#if 0 /* random write */
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
  /* seq write */
  cpu_time_t begin = rdtsc();

  unsigned long count = 0;
  
  for(auto& i: pages) {
    p[PAGE_SIZE*i] = 0xf;
    if(++count % 100000 == 0) PLOG("count: %lu", count);
  }

  cpu_time_t end = rdtsc();
  uint64_t usec = (end - begin) / 2400;
  PINF("%lu usec/page fault", usec / n_elements);
#endif
  
#if 1 /* seq write */  
  cpu_time_t begin = rdtsc();

  n_elements = 10;
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
  
  _pmem->free(p);
}
#endif

#if 0
TEST_F(Pmem_paged_test, IntegrityCheck)
{
  size_t n_elements = 10000;
  unsigned ITERATIONS = 1000000;
  uint64_t * p = nullptr;
  size_t slab_size = n_elements * sizeof(uint64_t);
  _pmem->allocate(slab_size,(void**)&p);

  ASSERT_FALSE(p==nullptr);

  /* zero */
  memset(p,0,slab_size);
  PLOG("Zeroing complete.");

  for(unsigned i=0;i<ITERATIONS;i++) {
    uint64_t slot = rand() % n_elements;
    if(p[slot] != 0 && p[slot] != slot) {
      PERR("Integrity check failed! slot-val=%ld expected=%ld or 0", p[slot], slot);
      ASSERT_TRUE(0);
    }
    p[slot] = slot;
    if(i % 100000 == 0) {
      PLOG("iterations: %lu", i);
    }
  }
    
  _pmem->free(p);
}

#endif

TEST_F(Pmem_paged_test, ReleaseBlockDevice)
{
  ASSERT_TRUE(_pmem);
  ASSERT_TRUE(_block);

  _pmem->stop();
  _pmem->release_ref();
  _pager->release_ref();
  _rm->release_ref();
  _block->release_ref();
}


} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
