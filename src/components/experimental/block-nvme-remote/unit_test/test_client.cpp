#include <gtest/gtest.h>
#include <string>
#include <common/cycles.h>
#include <common/logging.h>
#include <common/cpu.h>
#include <common/utils.h>

#include <component/base.h>
#include <api/block_itf.h>
#include <api/components.h>

using namespace std;
using namespace Component;

static char configFile[255];

namespace {

// The fixture for testing class Foo.
class Block_device_test : public ::testing::Test {

 protected:
  // You can remove any or all of the following functions if its body
  // is empty.

  Block_device_test() 
  {
  }

  virtual ~Block_device_test() {
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
  static IBlock_device* _client;
};


IBlock_device * Block_device_test::_client = NULL;

TEST_F(Block_device_test, Instantiation) {
  Component::IBase * comp = Component::load_component("../libcomanche-blknvme.so", Component::block_nvme_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");

  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  _client = fact->create(configFile);
  assert(_client);

  fact->release_ref();
}

#if 1
TEST_F(Block_device_test, Integrity)
{
  using namespace Component;
  
  io_buffer_t mem = _client->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  void * ptr = _client->virt_addr(mem);
  unsigned long ITERATIONS = 1000000;
  unsigned BLOCK_COUNT = 4000;

  /* zero blocks first */
  memset(ptr,0,4096);

  uint64_t tag;
  for(unsigned k=0;k < BLOCK_COUNT; k++) {
    tag = _client->async_write(mem, 0, k, 1);
  }
  while(!_client->check_completion(tag));
  
  PLOG("Zeroing complete OK.");

  
  for(unsigned long i=0;i<ITERATIONS;i++) {
    uint64_t lba = rand() % BLOCK_COUNT;
    
    /* read existing content */
    _client->read(mem, 0, lba, 1);
    
    uint64_t * v = (uint64_t*) ptr;
    if(v[0] != 0 && v[0] != lba) {
      PERR("value read from drive = %lx, expected %lx or 0", *v, lba);
      throw General_exception("bad data!");
    }
    /* TODO: check rest of block? */

    /* write out LBA into first 64bit */
    v[0] = lba;
    tag = _client->async_write(mem, 0, lba, 1);
    while(!_client->check_completion(tag));
    
    if(i % 1000 == 0) PLOG("Iteration: %lu",i);
  }

  PINF("Integrity check OK.");
}
#endif


#if 0
TEST_F(Block_device_test, WriteThroughput)
{
  using namespace Component;
  
  set_cpu_affinity(1UL << 2);

  sleep(1);
  
  io_buffer_t mem = _client->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);
  
  unsigned ITERATIONS = 1000000;
  uint64_t tags[ITERATIONS];

  /* warm up */
  for(unsigned i=0;i<100;i++) 
    tags[i] = _client->async_write(mem, 0, i, 1);
  while(!_client->check_completion(tags[99])); /* we only have to check the last completion */

  cpu_time_t start = rdtsc();

  for(unsigned i=0;i<ITERATIONS;i++) {
    tags[i] = _client->async_write(mem, 0, i, 1);
    //    PLOG("issued tag: %ld", tags[i]);
  }
  while(!_client->check_completion(tags[ITERATIONS-1]));

  cpu_time_t cycles_per_iop = (rdtsc() - start)/(ITERATIONS);
  PINF("took %ld cycles (%f usec) per IOP", cycles_per_iop,  cycles_per_iop / 2400.0f);
  PINF("rate: %f KIOPS", (2400.0 * 1000.0)/cycles_per_iop);

  _client->free_io_buffer(mem);
}
#endif

#if 1
TEST_F(Block_device_test, WriteLatency)
{
  set_cpu_affinity(1UL << 2);

  sleep(1);
  
  io_buffer_t mem = _client->allocate_io_buffer(4096,4096,Component::NUMA_NODE_ANY);

  unsigned ITERATIONS = 100;
  uint64_t tags[ITERATIONS];

  /* warm up */
  for(unsigned i=0;i<100;i++) 
    tags[i] = _client->async_write(mem, 0, i, 1);
  while(!_client->check_completion(tags[99])); /* we only have to check the last completion */

  for(unsigned i=0;i<ITERATIONS;i++) {
    cpu_time_t start = rdtsc();
    tags[i] = _client->async_write(mem, 0, i, 1);
    while(!_client->check_completion(tags[i]));
    cpu_time_t cycles_for_iop = rdtsc() - start;
    PINF("took %ld cycles (%f usec) per IOP", cycles_for_iop,  cycles_for_iop / 2400.0f);
  }

  _client->free_io_buffer(mem);
}
#endif

TEST_F(Block_device_test, Shutdown) {
  _client->release_ref();
}


}  // namespace

int main(int argc, char **argv) {
  if(argc != 2) {
    printf("test_client <.json config>\n");
    return -1;
  }
  
  strcpy(configFile, argv[1]);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
