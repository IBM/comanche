#include "region_modifications.h"
#include "allocator_ra.h"

#include <gtest/gtest.h>

//#define GPERF_TOOLS

#ifdef GPERF_TOOLS
#include <gperftools/profiler.h>
#endif

// The fixture for testing class Libnupm.
class Libnupm_test : public ::testing::Test {
 protected:
  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp()
  {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown()
  {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Objects declared here can be used by all tests in the test case
};

TEST_F(Libnupm_test, RegionModifications)
{
  struct {
    char c = 0;
    /* padding here */
    int i = 2;
    int j = 2;
  } z;
  nupm::region_tracker_add(&z.j, sizeof z.j);
  nupm::region_tracker_add(&z.c, sizeof z.c);
  nupm::region_tracker_add(&z.i, sizeof z.i);

  const void *v;
  std::size_t sz;
  sz = nupm::region_tracker_get_region(1, v);
  EXPECT_EQ(sizeof z.i + sizeof z.j, sz);
  EXPECT_EQ(&z.i, v);
  sz = nupm::region_tracker_get_region(0, v);
  EXPECT_EQ(sizeof z.c, sz);
  EXPECT_EQ(&z.c, v);
  /* past last element */
  sz = nupm::region_tracker_get_region(3, v);
  EXPECT_EQ(0, sz);
  /* farther past last element */
  sz = nupm::region_tracker_get_region(5, v);
  EXPECT_EQ(0, sz);
  /* before first element: also nothing */
  sz = nupm::region_tracker_get_region(-1, v);
  EXPECT_EQ(0, sz);
  nupm::region_tracker_coalesce_across_TLS();
  sz = nupm::region_tracker_get_region(1, v);
  EXPECT_EQ(sizeof z.i + sizeof z.j, sz);
  EXPECT_EQ(&z.i, v);
  nupm::region_tracker_clear();
  sz = nupm::region_tracker_get_region(0, v);
  EXPECT_EQ(0, sz);
}

TEST_F(Libnupm_test, AVL_allocator)
{
  auto size = 1000000;
  void *v = malloc(size);
  ASSERT_NE(nullptr, v);
  /* AVL_range_allocator requires an addr_t, defined in comanche common/types.h */
  Core::AVL_range_allocator ra(reinterpret_cast<addr_t>(v), size);
  auto a = nupm::allocator_ra<char>(ra);
  const std::size_t ITERATIONS = 100;
  PLOG("AVL_allocator running... %zu allocations", ITERATIONS);

  std::vector<char *> alloc_v;
  alloc_v.reserve(ITERATIONS);

  auto start = std::chrono::high_resolution_clock::now();
  constexpr auto alloc_size = 200;
  for (unsigned i = 0; i < ITERATIONS; i++) {
    auto p = a.allocate(alloc_size);
    ASSERT_TRUE(p);
    EXPECT_LE(v, p);
    EXPECT_LT(p + alloc_size, static_cast<char *>(v) + size);
    alloc_v.push_back(p);
  }

  auto end = std::chrono::high_resolution_clock::now();
  double secs = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
  PINF("AVL_allocator: %lu allocs/sec",
       static_cast<unsigned long>( double(ITERATIONS) / secs)
  );

  PLOG("AVL_allocator: Freeing %zu allocations...", alloc_v.size());
  for (auto &vv : alloc_v) {
    a.deallocate(vv, alloc_size);
  }
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  auto r = RUN_ALL_TESTS();

  return r;
}
