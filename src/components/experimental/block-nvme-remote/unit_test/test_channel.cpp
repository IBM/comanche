#include <gtest/gtest.h>
#include <string>
#include <common/cycles.h>
#include "../src/volume_agent.h"
#include <eal_init.h>

using namespace std;
using namespace comanche;

namespace {

// The fixture for testing class Foo.
class Channel_test : public ::testing::Test {

 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  Channel_test() 
  {
    // You can do set-up work for each test here.
    if(!_ch)
      _ch = new Channel(3/* core */);
  }

  virtual ~Channel_test() {
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
  static Channel* _ch;
};

std::string server_name;
Channel * Channel_test::_ch = NULL;

TEST_F(Channel_test, ChannelConnect) {
  if(server_name.empty())
    _ch->wait_for_connect();
  else
    _ch->connect(server_name.c_str());
}

TEST_F(Channel_test, Send) {
  Buffer_manager buffers("test-channel",*_ch, 256, 4096);

  const size_t ITERATIONS = 1000000;
  size_t TO_SEND = ITERATIONS;

  
  if(server_name.empty()) {
    while(TO_SEND > 0) {
      while(!buffers.empty()) {
        struct ibv_mr * mr = buffers.alloc();
        //        PLOG("alloc MR: %p", mr);
        _ch->post_recv(reinterpret_cast<uint64_t>(mr), mr);
      }
      TO_SEND -= _ch->poll_completions([&buffers](uint64_t wid){
          //PNOTICE("free MR: %p", buffer);
          buffers.free((struct ibv_mr*)wid);
        });
    }
    PLOG("recv OK");
  }
  else {
    cpu_time_t start = rdtsc();
    
    while(TO_SEND > 0) {
      while(!buffers.empty()) {
        struct ibv_mr * mr = buffers.alloc();
        _ch->post_send((uint64_t) mr, mr);
        TO_SEND--;
        if(TO_SEND==0) break;
      }
      _ch->poll_completions([&buffers](uint64_t wid){
          //PLOG("send completion");
          buffers.free((struct ibv_mr*)wid);
        });
      if(TO_SEND==0) break;
    }
    PLOG("send OK (%lu)", TO_SEND);
    
    cpu_time_t end = rdtsc();
    float usec = (end-start)/2400.0;
    float sec = usec/1000000.0;
    PLOG("time: %f usec", usec);
    unsigned long megabytes = (4 * ITERATIONS)/1024.0f;
    PLOG("throughput %f MB/s", (1.0/sec)*((float)megabytes));
    PLOG("%ld cycles %f cycles per IOP", end-start, (end-start)/((float)ITERATIONS));
  }
}

TEST_F(Channel_test, Shutdown) {  delete _ch; }


}  // namespace

int main(int argc, char **argv) {

  DPDK::eal_init(64); // TODO

  if(argc > 1) {
    server_name = argv[1];
  }
  
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
