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
#include <core/postbox.h>

//#define TEST_UIPC
#define TEST_POSTBOX

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

#ifdef TEST_UIPC
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
#endif

#ifdef TEST_POSTBOX
TEST_F(Core_test, Postbox)
{
#define CHECK
#ifdef CHECK
  unsigned long ITERATIONS = 1000;
#else
  unsigned long ITERATIONS = 10000000;
#endif
  if(fork()) { /* fork */
    Core::UIPC::Shared_memory sm("zimbar", 1);
    sleep(1);
    Core::Mpmc_postbox<uint64_t> pbox(sm.get_addr(), sizeof(uint64_t) * 64, true);

    auto started = std::chrono::high_resolution_clock::now();
    for(uint64_t i=0;i<ITERATIONS;i++) {
      if(pbox.post(i+1)) {
        PLOG("posted %lu", i+1);
      }
      else {
        i--;
      }
    }

#ifndef CHECK
    auto done = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(done-started).count();
    auto secs = ((float)ms)/1000.0f;
    PINF("Duration %f seconds", secs);
    PINF("Rate: %f M items per second", (ITERATIONS / secs)/1000000.0);
#endif
  }
  else {
    usleep(100000);
    Core::UIPC::Shared_memory sm("zimbar");
    Core::Mpmc_postbox<uint64_t> pbox(sm.get_addr(), sizeof(uint64_t) * 64);
#ifdef CHECK
    std::list<uint64_t> collected;
#endif
    for(uint64_t i=0;i<ITERATIONS;i++) {
      uint64_t val = 0;
      if(pbox.collect(val)) {
        //        PLOG("collected %lu", val);
#ifdef CHECK
        collected.push_back(val);
#endif
      }
      else {
        i--;
      }      
    }

#ifdef CHECK
    collected.sort();
    uint64_t expect = 1;
    for(auto& e: collected) {
      ASSERT_TRUE(e == expect);
      expect++;
    }
#endif
  }
  int status;
  wait(&status);
  
}
#endif


} // namespace

int main(int argc, char **argv) {

  if(argc > 1) client_side = true;
  
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
