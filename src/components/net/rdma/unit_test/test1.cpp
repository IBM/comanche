/* note: we do not include component source, only the API definition */
#include <gtest/gtest.h>
#include <api/components.h>
#include <api/rdma_itf.h>
#include <common/utils.h>
#include <core/dpdk.h>
#include <core/physical_memory.h>

using namespace Component;

namespace {

// The fixture for testing class Foo.
class Rdma_test : public ::testing::Test {

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
  static Component::IRdma * _rdma;
};

bool client;
Component::IRdma * Rdma_test::_rdma;

TEST_F(Rdma_test,DpdkInit)
{
  if(client)
    DPDK::eal_init(32, 0, false); /* slave */
  else
    DPDK::eal_init(32, 0, true); /* master */
}

TEST_F(Rdma_test, Instantiate)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-rdma.so",
                                                      Component::net_rdma_factory);

  ASSERT_TRUE(comp);
  IRdma_factory * fact = (IRdma_factory *) comp->query_interface(IRdma_factory::iid());

  _rdma = fact->create("any");
  
  fact->release_ref();
}

int port = 18515;
std::string remote_host;

TEST_F(Rdma_test, Connect)
{
  if(client) {
    ASSERT_TRUE(_rdma->connect(remote_host.c_str(), port) == S_OK);
  }
  else {
    ASSERT_TRUE(_rdma->wait_for_connect(port) == S_OK);
  }
  PLOG("Connected OK!");
}

TEST_F(Rdma_test, ExchangeData)
{
  Core::Physical_memory mem;
  Component::io_buffer_t iob = mem.allocate_io_buffer(KB(4),
                                                      KB(4),
                                                      Component::NUMA_NODE_ANY);
  unsigned ITERATIONS = 100;

  char  * msg = (char *) mem.virt_addr(iob);
  auto rdma_buffer = _rdma->register_memory(mem.virt_addr(iob),KB(4));

  for(unsigned i=0;i<ITERATIONS;i++) {
    if(client) {
      sprintf(msg, "Hello %u !!", i);
      _rdma->post_send(i, rdma_buffer);
      int n_complete = 0;
      while((n_complete = _rdma->poll_completions()) == 0);
      ASSERT_TRUE(n_complete == 1);
    }
    else {
      _rdma->post_recv(i, rdma_buffer);
      int n_complete = 0;
      while((n_complete = _rdma->poll_completions()) == 0);
      ASSERT_TRUE(n_complete == 1);
      PLOG("received: %s", msg);
    }
  }
  
  mem.free_io_buffer(iob);
}


TEST_F(Rdma_test, Cleanup)
{
  _rdma->release_ref();
}



} // namespace

int main(int argc, char **argv) {

  if(argc < 2) {
    PINF("rdma-test1 [client ipaddr| server]");
    return -1;
  }

  client = (strcmp(argv[1],"client")==0);
  if(client) {
    assert(argc==3);
    remote_host = argv[2];
  }  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
