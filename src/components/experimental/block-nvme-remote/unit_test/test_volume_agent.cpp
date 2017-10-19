#include <gtest/gtest.h>
#include <string>
#include <common/cycles.h>
#include "../src/volume_agent.h"
#include "../src/volume_agent_session.h"

using namespace std;
using namespace comanche;

namespace {

// The fixture for testing class Foo.
class Volume_agent_test : public ::testing::Test {

 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  Volume_agent_test() 
  {
    // You can do set-up work for each test here.
    if(!_va)
      _va = new Volume_agent("test-vol-config.json");

    if(!_sessions[0]) {
      _sessions[0] = _va->create_session(0);
    }
  }

  virtual ~Volume_agent_test() {
    // You can do clean-up work that doesn't throw exceptions here.
  }

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
  static Volume_agent* _va;
  static Volume_agent_session* _sessions[2];
};

Volume_agent_session* Volume_agent_test::_sessions[2] = {NULL,NULL};
Volume_agent * Volume_agent_test::_va = NULL;

TEST_F(Volume_agent_test, VolumeAgentInstantiation) {}

#if 0
TEST_F(Volume_agent_test, SyncWriteIO)
{
  unsigned BLOCK_SIZE = 4096;
  unsigned NUM_BLOCKS = 100;
  
  void * ptr = aligned_alloc(BLOCK_SIZE, BLOCK_SIZE);
  EXPECT_TRUE(ptr);

  PLOG("Starting traffic send!!!");
  channel_memory_t mr = _sessions[0]->register_region(ptr, BLOCK_SIZE);

  cpu_time_t start;
  for(unsigned i=0;i<NUM_BLOCKS;i++) {
    if(i == 10) start=rdtsc();
    *((uint64_t *)ptr) = i;
    cpu_time_t begin;
    do {
      begin = rdtsc();
    }
    while(_sessions[0]->submit_sync_op(mr, i, 1, comanche::COMANCHE_OP_WRITE)!=S_OK);
    PINF("time=%f usec", ((float)(rdtsc() - begin))/2400.0);
  }
  cpu_time_t cycles_per_iop = (rdtsc() - start)/NUM_BLOCKS;
  PINF("took %ld cycles (%ld usec) per IOP", cycles_per_iop,  cycles_per_iop / 2400);
  
  free(ptr);
}
#endif

#if 1
TEST_F(Volume_agent_test, AsyncWriteIO)
{
  unsigned BLOCK_SIZE = 4096;
  unsigned NUM_BLOCKS = 1000;
  unsigned ITERATIONS = 1000;

  assert(_sessions[0]);
  
  channel_memory_t mr[NUM_BLOCKS];
  for(unsigned i=0;i<NUM_BLOCKS;i++) {
    mr[i] = _sessions[0]->alloc_region(BLOCK_SIZE, BLOCK_SIZE);
    memset(mr[i]->addr, i, BLOCK_SIZE);
    EXPECT_TRUE(mr[i]);
  }

  //uint64_t last_gwid = 0;
  
  cpu_time_t start = rdtsc();

  for(unsigned i=0;i<ITERATIONS;i++) {
    unsigned submits = 0;
    
    while(submits < NUM_BLOCKS) {

      _sessions[0]->submit_async(mr[submits],
                                 submits /* lba */,
                                 1, COMANCHE_OP_WRITE);
      submits++;
      if(submits >= NUM_BLOCKS) break;
    }

    while(_sessions[0]->poll_outstanding() > 0);
  }

  for(unsigned i=0;i<NUM_BLOCKS;i++) {
    _sessions[0]->free_region(mr[i]);
  }
  
  cpu_time_t delta_t = rdtsc() - start;
  cpu_time_t cycles_per_iop = (delta_t)/ (NUM_BLOCKS*ITERATIONS);
  PINF("total microseconds: %ld", delta_t / 2400);
  PINF("took %ld cycles (%f usec) per IOP",
       cycles_per_iop,  cycles_per_iop / 2400.0f);  
}
#endif

TEST_F(Volume_agent_test, MultisessionAsyncWriteIO)
{
  unsigned BLOCK_SIZE = 4096;
  unsigned NUM_BLOCKS = 1000;
  unsigned ITERATIONS = 1000;
  
  assert(_sessions[0]);

  if(!_sessions[1])
    _sessions[1] = _va->create_session(1);
  
  channel_memory_t mr[NUM_BLOCKS];
  for(unsigned i=0;i<NUM_BLOCKS;i++) {
    mr[i] = _sessions[0]->alloc_region(BLOCK_SIZE, BLOCK_SIZE);
    memset(mr[i]->addr, i, BLOCK_SIZE);
    EXPECT_TRUE(mr[i]);
  }

  channel_memory_t mr1[NUM_BLOCKS];
  for(unsigned i=0;i<NUM_BLOCKS;i++) {
    mr1[i] = _sessions[1]->alloc_region(BLOCK_SIZE, BLOCK_SIZE);
    memset(mr1[i]->addr, i, BLOCK_SIZE);
    EXPECT_TRUE(mr1[i]);
  }

  cpu_time_t start = rdtsc();

  for(unsigned i=0;i<ITERATIONS;i++) {
    unsigned submits = 0;
    
    while(submits < NUM_BLOCKS) {

      _sessions[0]->submit_async(mr[submits],
                                  submits /* lba */,
                                  1,
                                 COMANCHE_OP_WRITE);

      _sessions[1]->submit_async(mr1[submits],
                                 2*submits /* lba */,
                                 1,
                                 COMANCHE_OP_WRITE);

      submits++;
      if(submits >= NUM_BLOCKS) break;
    }

    while(_sessions[0]->poll_outstanding() > 0);
    while(_sessions[1]->poll_outstanding() > 0);
  }

  for(unsigned i=0;i<NUM_BLOCKS;i++) {
    _sessions[0]->free_region(mr[i]);
    _sessions[1]->free_region(mr1[i]);
  }
  
  cpu_time_t delta_t = rdtsc() - start;
  cpu_time_t cycles_per_iop = (delta_t)/ (2*NUM_BLOCKS*ITERATIONS);
  
  PINF("total microseconds: %ld", delta_t / 2400);
  PINF("took %ld cycles (%f usec) per IOP",
       cycles_per_iop,  cycles_per_iop / 2400.0f);  

}

TEST_F(Volume_agent_test, Shutdown) {
  delete _va; /* this will delete sessions too */
}

// // Tests that the Foo::Bar() method does Abc.
// TEST_F(Volume_agent_test, MethodBarDoesAbc) {
//   const string input_filepath = "this/package/testdata/myinputfile.dat";
//   const string output_filepath = "this/package/testdata/myoutputfile.dat";
//   Foo f;
//   EXPECT_EQ(0, f.Bar(input_filepath, output_filepath));
// }

// // Tests that Foo does Xyz.
// TEST_F(Volume_agent_test, DoesXyz) {
//   // Exercises the Xyz feature of Foo.
// }

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
