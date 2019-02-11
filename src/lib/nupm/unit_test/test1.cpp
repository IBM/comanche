/* note: we do not include component source, only the API definition */
#include <chrono>
#include <gtest/gtest.h>
#include <common/utils.h>
#include <common/rand.h>
#include <common/cycles.h>
#include <core/heap_allocator.h>
#include "tx_cache.h"
#include "vmem_numa.h"
#include "arena_alloc.h"
#include "rc_alloc_avl.h"
#include "rc_alloc_lb.h"
#include "dax_map.h"

//#define GPERF_TOOLS

#ifdef GPERF_TOOLS
#include <gperftools/profiler.h>
#endif

struct
{
  uint64_t uuid;
} Options;

// The fixture for testing class Foo.
class Libnupm_test : public ::testing::Test {

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
};


// static bool memcheck(void * p, char val, size_t len)
// {
//   for(size_t i=0; i<len; i++) {
//     if(((char *)p)[i] != val)
//       return false;
//   }
//   return true;
// }

//#define RUN_RPALLOCATOR_TESTS
//#define RUN_VMEM_ALLOCATOR_TESTS
//#define RUN_DEVDAX_TEST
//#define RUN_AVL_RCA_TEST
//#define RUN_AVL_STRESS_TEST
//#define RUN_MALLOC_STRESS_TEST
//#define RUN_LB_TEST
//#define RUN_LB_STRESS_TEST
#define RUN_LB_INTEGRITY_TEST



#ifdef RUN_LB_INTEGRITY_TEST
TEST_F(Libnupm_test, RcAllocatorLBIntegrity)
{
  const size_t ARENA_SIZE = GB(32);
  void * p = aligned_alloc(GB(1), ARENA_SIZE);
  ASSERT_TRUE(p);

  nupm::Rca_LB rca;
  rca.add_managed_region(p, ARENA_SIZE, 0);

  std::vector<iovec> log;
 
  /* populate with 1M entries */
  const size_t COUNT = 10;
  for(size_t i=1; i<COUNT; i++) {
    size_t s = MiB(2)*i; //1024; // ((genrand64_int64() % 1024) + 1) * 8;
    assert(s % 8 == 0);
    void * p = rca.alloc(s, 0 /* numa */, 8 /* alignment */);
    log.push_back({p,s});
  }



  free(p);
}
#endif



#ifdef RUN_LB_STRESS_TEST
TEST_F(Libnupm_test, RcAllocatorLBStress)
{
  const size_t ARENA_SIZE = GB(32);
  void * p = aligned_alloc(GB(1), ARENA_SIZE);
  ASSERT_TRUE(p);

  nupm::Rca_LB rca;
  rca.add_managed_region(p, ARENA_SIZE, 0);

#ifdef GPERF_TOOLS
  ProfilerStart("cpu_profile");
#endif
 
  __sync_synchronize();
  auto start_time = std::chrono::high_resolution_clock::now();
  /* populate with 1M entries */
  const size_t COUNT = 10000000;
  for(size_t i=0; i<COUNT; i++) {
    size_t s = 8; // (genrand64_int64() % 256) + 8;
    rca.alloc(s, 0 /* numa */, 0 /* alignment */);
  }
  __sync_synchronize();

#ifdef GPERF_TOOLS
  ProfilerStop();
#endif

  auto end_time = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
  double per_sec = (((double)COUNT)/secs);
  PINF("Non-aligned, 64 byte size, insertion rate: %.0fK /sec (mean latency:%.0f usec)", per_sec/1000.0, 1000000.0/per_sec);

  free(p);
}
#endif

#ifdef RUN_LB_TEST
TEST_F(Libnupm_test, RcAllocatorLB)
{
  void * p = aligned_alloc(GB(1),GB(8));
  ASSERT_TRUE(p);
  void * q = aligned_alloc(GB(1),GB(8));
  ASSERT_TRUE(q);

  PLOG("p=%p", p);
  PLOG("q=%p", q);

  std::vector<iovec> allocations;

  std::string state_A, state_B;
  { /* create allocator */
    nupm::Rca_LB rca;
    rca.add_managed_region(p, GB(8), 0);
    rca.add_managed_region(q, GB(8), 1);

    for(unsigned i=0;i<10;i++) {
      void * a = rca.alloc(64, 0, 16);
      ASSERT_TRUE(a != nullptr);
      PLOG("Allocated: %p", a);
      allocations.push_back({a, (i+1)*32});
    }

    /* logical power-fail */
    rca.debug_dump(&state_A);// TODO get string version

    /* now we should be able to free */
    for(auto& i: allocations) {
      rca.free(i.iov_base, 0);
      PLOG("Freed: %p", i.iov_base);
    }

  }

  // {
  //   nupm::Rca_LB rca;
  //   rca.add_managed_region(p, GB(8), 0);
  //   rca.add_managed_region(q, GB(8), 1);

  //   for(auto& i: allocations) {
  //     rca.inject_allocation(i.iov_base, i.iov_len, 0);
  //   }
    
  //   rca.debug_dump(&state_B);

  //   /* state A and B should be the same */
  //   ASSERT_TRUE(state_A == state_B);
    
  //   /* now we should be able to free */
  //   for(auto& i: allocations) {
  //     rca.free(i.iov_base, 0);
  //   }
  // }

  free(p);
  free(q);
}
#endif


#ifdef RUN_DEVDAX_TEST
TEST_F(Libnupm_test, DevdaxManager)
{
  {
    nupm::Devdax_manager ddm(true);
  }
  nupm::Devdax_manager ddm; // rebuild

  size_t p_len = 0;
  uint64_t uuid = Options.uuid;
  assert(uuid > 0);
  size_t size = GB(2);
  ddm.debug_dump(0);

  PLOG("Opening existing region..");
  void * p = ddm.open_region(uuid, 0, &p_len);
  if(p) {
    PLOG("Opened existing region %p OK", p);
    ddm.debug_dump(0);
    PLOG("Now erasing it...");
    ddm.erase_region(uuid, 0);
    ddm.debug_dump(0);
  }

  p = ddm.create_region(uuid, 0, size);
  if(p)
    PLOG("created region %p ", p);
  ASSERT_TRUE(p);
  memset(p, 0, 4096);
  ddm.debug_dump(0);

  PLOG("Erase...");
  ddm.erase_region(uuid, 0);
  ddm.debug_dump(0);

  PLOG("Re-create...");
  void * q = ddm.create_region(uuid, 0, size);
  ASSERT_TRUE(q == p);

  ddm.debug_dump(0);
  ddm.erase_region(uuid, 0);
}
#endif

#ifdef RUN_AVL_RCA_TEST
TEST_F(Libnupm_test, RcAllocatorAVL)
{
  void * p = aligned_alloc(GB(1),GB(8));
  ASSERT_TRUE(p);
  void * q = aligned_alloc(GB(1),GB(8));
  ASSERT_TRUE(q);

  PLOG("p=%p", p);
  PLOG("q=%p", q);

  std::vector<iovec> allocations;

  std::string state_A, state_B;
  { /* create allocator */
    nupm::Rca_AVL rca;
    rca.add_managed_region(p, GB(8), 0);
    rca.add_managed_region(q, GB(8), 1);

    for(unsigned i=0;i<10;i++) {
      void * a = rca.alloc((i+1)*32, 0, 16);
      allocations.push_back({a, (i+1)*32});
    }

    /* logical power-fail */
    rca.debug_dump(&state_A);// TODO get string version 
  }

  {
    nupm::Rca_AVL rca;
    rca.add_managed_region(p, GB(8), 0);
    rca.add_managed_region(q, GB(8), 1);

    for(auto& i: allocations) {
      rca.inject_allocation(i.iov_base, i.iov_len, 0);
    }
    
    rca.debug_dump(&state_B);

    /* state A and B should be the same */
    ASSERT_TRUE(state_A == state_B);
    
    /* now we should be able to free */
    for(auto& i: allocations) {
      rca.free(i.iov_base, 0);
    }
  }

  free(p);
  free(q);
}
#endif


#ifdef RUN_AVL_STRESS_TEST
TEST_F(Libnupm_test, RcAllocatorStress)
{
  const size_t ARENA_SIZE = GB(32);
  void * p = aligned_alloc(GB(1), ARENA_SIZE);
  ASSERT_TRUE(p);

  nupm::Rca_AVL rca;
  rca.add_managed_region(p, ARENA_SIZE, 0);
  
  __sync_synchronize();
  auto start_time = std::chrono::high_resolution_clock::now();
  /* populate with 1M entries */
  const size_t COUNT = 10000000;
  for(size_t i=0; i<COUNT; i++) {
    size_t s = 32; //(genrand64_int64() % 32) + 1;
    rca.alloc(s, 0 /* numa */, 0 /* alignment */);
  }
  __sync_synchronize();
  auto end_time = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
  double per_sec = (((double)COUNT)/secs);
  PINF("Non-aligned, 32 byte size, insertion rate: %.0fK /sec (mean latency:%.0f usec)", per_sec/1000.0, 1000000.0/per_sec);

  free(p);
}
#endif

#ifdef RUN_AVL_STRESS_TEST_MEMKIND
TEST_F(Libnupm_test, RcAllocatorStressMemkind)
{
  const size_t ARENA_SIZE = GB(32);
  void * p = aligned_alloc(GB(1), ARENA_SIZE);
  ASSERT_TRUE(p);

  nupm::Rca_AVL rca;
  rca.add_managed_region("/dev/dax0.3",0);
  
  //  std::vector<iovec> allocations;
  //  allocations.push_back({a, s});
  __sync_synchronize();
  auto start_time = std::chrono::high_resolution_clock::now();
  /* populate with 1M entries */
  const size_t COUNT = 10000000;
  for(size_t i=0; i<COUNT; i++) {
    size_t s = 32; //(genrand64_int64() % 32) + 1;
    rca.alloc(s, 0 /* numa */, 0 /* alignment */);
  }
  __sync_synchronize();
  auto end_time = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
  double per_sec = (((double)COUNT)/secs);
  PINF("Non-aligned, 32 byte size, insertion rate: %.0fK /sec (mean latency:%.0f usec)", per_sec/1000.0, 1000000.0/per_sec);

  free(p);
}
#endif

#ifdef RUN_MALLOC_STRESS_TEST
TEST_F(Libnupm_test, MallocStress)
{
  //  std::vector<iovec> allocations;
  //  allocations.push_back({a, s});
  __sync_synchronize();
  auto start_time = std::chrono::high_resolution_clock::now();
  /* populate with 1M entries */
  const size_t COUNT = 10000000;
  for(size_t i=0; i<COUNT; i++) {
    size_t s = 32; //(genrand64_int64() % 32) + 1;
    void * p = ::malloc(s);
    assert(p);
  }
  __sync_synchronize();
  auto end_time = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
  PLOG("malloc: secs %.2f", secs);
  double per_sec = (((double)COUNT)/secs);
  PINF("malloc: Non-aligned, 32 byte size, insertion rate: %.2fM /sec", per_sec/1000000.0);
}
#endif


#if 0
TEST_F(Libnupm_test, NdControl)
{
  nupm::ND_control ctrl;
  PLOG("ND_control init complete!");
}
#endif

#if 0
TEST_F(Libnupm_test, TxCache)
{
  size_t NUM_PAGES = 2048;
  size_t p_size = NUM_PAGES * MB(2);
  void * p = nupm::allocate_virtual_pages(p_size/MB(2), MB(2), 0x900000000ULL);

  cpu_time_t start = rdtsc();
  for(size_t i=0;i<NUM_PAGES;i++) {
    ((char*)p)[i*MB(2)]='a';
  }
  cpu_time_t delta = (rdtsc() - start)/NUM_PAGES;
  PLOG("Mean PF cost: (%f usec) %lu", Common::cycles_to_usec(delta), delta);  
  // { // important
  //   Core::Heap_allocator<char> heap(p, p_size, "heapA");

  //   char * q = heap.allocate(128);
  //   memset(q, 0, 128);
  // }

#if 0
  char * c = (char*) p;
  cpu_time_t start = rdtsc();
  c[0] = 'a';
  cpu_time_t delta = rdtsc() - start;
  c[4096] = 'a';
  PLOG("touched! in %ld cycles (%f usec)", delta, Common::cycles_to_usec(delta));
#endif
  
  nupm::free_virtual_pages(p);
}
#endif

#ifdef RUN_RPALLOCATOR_TESTS
TEST_F(Libnupm_test, RpAllocatorPerf)
{
  nupm::Rp_allocator_volatile<1> allocator;
  allocator.initialize_thread();

  std::chrono::system_clock::time_point start, end;
  
  const unsigned ITERATIONS = 10000000;
  std::vector<void*> alloc_v;
  alloc_v.reserve(ITERATIONS);

  start = std::chrono::high_resolution_clock::now();
  for(unsigned i=0;i<ITERATIONS;i++) {
    void * p = allocator.alloc(64, i % 2);
    ASSERT_TRUE(p);
    alloc_v.push_back(p);
  }
  end = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PINF("Arena::SingleThreadAlloc: %lu allocs/sec",
       (unsigned long) (((double) ITERATIONS) / secs));

  for(auto& a : alloc_v)
    allocator.free(a);
}

TEST_F(Libnupm_test, RpAllocatorPerf2)
{
  nupm::Rp_allocator_volatile<0> allocator;
  allocator.initialize_thread();

  std::chrono::system_clock::time_point start, end;
  
  const unsigned ITERATIONS = 10000000;
  std::vector<void*> alloc_v;
  alloc_v.reserve(ITERATIONS);

  start = std::chrono::high_resolution_clock::now();
  for(unsigned i=0;i<ITERATIONS;i++) {
    size_t s = genrand64_int64() % 8096;
    void * p = allocator.alloc(s, i % 2);
    ASSERT_TRUE(p);
    alloc_v.push_back(p);
  }
  end = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PINF("Arena::SingleThreadAlloc: %lu allocs/sec",
       (unsigned long) (((double) ITERATIONS) / secs));

  for(auto& a : alloc_v)
    allocator.free(a);
}


TEST_F(Libnupm_test, RpAllocatorIntegrity)
{
  nupm::Rp_allocator_volatile<0> allocator;
  allocator.initialize_thread();

  const unsigned ITERATIONS = 1000000;
  std::vector<void*> alloc_v;
  alloc_v.reserve(ITERATIONS);

  for(unsigned i=0;i<ITERATIONS;i++) {
    void * p = allocator.alloc(1024, i % 2);
    ASSERT_TRUE(p);
    alloc_v.push_back(p);
    char m = ((char*)p)[7];
    memset(p, m, 1024);
  }

  for(auto& a : alloc_v) {
    char m = ((char*)a)[7];
    ASSERT_TRUE(memcheck(a,m,1024));
    allocator.free(a);
  }
}
#endif

#if 0
TEST_F(Libnupm_test, Arena)
{
  nupm::Arena_allocator_volatile allocator;

  std::chrono::system_clock::time_point start, end;
  PLOG("SingleThreadAlloc running...");
  
  const unsigned ITERATIONS = 3000;
  std::vector<void*> alloc_v;
  alloc_v.reserve(ITERATIONS);

  start = std::chrono::high_resolution_clock::now();
  for(unsigned i=0;i<ITERATIONS;i++) {
    void * p = allocator.alloc(i % 2);
    ASSERT_TRUE(p);
    alloc_v.push_back(p);
    //memset(p, 0xe, s);
    ((char*)p)[0] = 0xe;
  }
  end = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PINF("Arena::SingleThreadAlloc: %lu allocs/sec", (unsigned long) (((double) ITERATIONS) / secs));

  for(auto& a : alloc_v) {
    allocator.free(a);
  }
}
#endif

#if 0
TEST_F(Libnupm_test, Basic)
{
  nupm::Vmem_allocator allocator;

  void * ptr0 = allocator.alloc(0, MB(2));
  PLOG("ptr0: %p", ptr0);
  ASSERT_TRUE(ptr0);
  memset(ptr0, 0xe, MB(2));
  void * ptr1 = allocator.alloc(1, MB(2));
  ASSERT_TRUE(ptr1);
  PLOG("ptr1: %p", ptr1);
  memset(ptr1, 0xe, MB(2));

  allocator.free(ptr0);
  allocator.free(ptr1);
  PLOG("frees OK!");
}
#endif

#ifdef RUN_VMEM_ALLOCATOR_TESTS

TEST_F(Libnupm_test, VmemSingleThreadAlloc)
{
  std::chrono::system_clock::time_point start, end;
  nupm::Vmem_allocator allocator;
  PLOG("Vmem::SingleThreadAlloc running...");
  
  const unsigned ITERATIONS = 10000000;
  std::vector<void*> alloc_v;
  alloc_v.reserve(ITERATIONS);

  start = std::chrono::high_resolution_clock::now();
  for(unsigned i=0;i<ITERATIONS;i++) {
    size_t s = genrand64_int64() % 8096;
    void * p = allocator.alloc(i % 2 /* numa zone */, s);
    ASSERT_TRUE(p);
    alloc_v.push_back(p);
    //memset(p, 0xe, s);
    //    ((char*)p)[0] = 0xe;
  }
  end = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  PINF("Vmem::SingleThreadAlloc: %lu allocs/sec", (unsigned long) (((double) ITERATIONS) / secs));

  PLOG("Vmem::SingleThreadAlloc: Freeing allocotions...");
  for(auto& a : alloc_v) {
    allocator.free(a);
  }
}

#endif


int main(int argc, char **argv) {
  
  ::testing::InitGoogleTest(&argc, argv);

  if(argc > 1) {
    Options.uuid = atol(argv[1]);
  }
  auto r = RUN_ALL_TESTS();

  return r;
}
