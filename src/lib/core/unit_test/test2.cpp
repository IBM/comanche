#include <gtest/gtest.h>
#include <string>
#include <unistd.h>
#include <common/cycles.h>
#include <common/utils.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/cpu.h>
#include <core/conc_avl_tree.h>
#include <core/dpdk.h>
#include <core/physical_memory.h>
#include <core/uipc.h>

namespace {

// The fixture for testing class Foo.
class Core_test : public ::testing::Test {

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

bool client_side = false;

TEST_F(Core_test, UIPC)
{
  unsigned long ITERATIONS = 10000000;
  //  if(!client_side) {
  if(fork()) {
    Core::UIPC::Channel sm("foobar2", 4096, 32);
    
    for(unsigned long i=0;i<ITERATIONS;i++) {
      void * msg = sm.alloc_msg();
      while(sm.send(msg) != S_OK) {
        //        usleep(1000);
      }
      if(i % 1000000 == 0) PLOG("sent %lu", i);
    }
    PLOG("%lu sent", ITERATIONS);
  }
  else {
    Core::UIPC::Channel sm("foobar2");
    
    void * incoming = nullptr;

    auto started = std::chrono::high_resolution_clock::now();

    for(unsigned long i=0;i<ITERATIONS;i++) {
      incoming = nullptr;
      while(sm.recv(incoming)==E_EMPTY);
      sm.free_msg(incoming);
      if(i % 1000000 == 0) PLOG("sent %lu", i);
    }
    PLOG("%lu received", ITERATIONS);
    auto done = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();
    auto secs = ((float)ms)/1000.0f;
    PINF("Duration %f seconds", secs);
    PINF("Rate: %f M items per second", (ITERATIONS / secs)/1000000.0);

  }
  void* status;
  wait(&status);
}


} // namespace

int main(int argc, char **argv) {

  if(argc > 1) client_side = true;
  
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
