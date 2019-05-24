#include "region_modifications.h"

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

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  auto r = RUN_ALL_TESTS();

  return r;
}
