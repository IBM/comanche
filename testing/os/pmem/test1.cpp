#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <string>
#include <set>
#include <list>
#include <vector>
#include <algorithm>
#include <libpmemobj.h>
#include <libpmem.h>

#include <common/logging.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <common/cycles.h>
#include <common/rand.h>
#include <common/cpu.h>

#include "pmem_helper.h"

//#define LAYOUT_NAME "intro_0" /* will use this in create and open */
#define MAX_BUF_LEN 10 /* maximum length of our buffer */

#define RAND genrand64_int64


class Byte_8_rw_test : public Pmem_base
{
  static constexpr unsigned DATASIZE = MB(8);
public:
  Byte_8_rw_test(std::string filename, size_t iterations) : Pmem_base(filename, MB(8)), _size(MB(8)) {
    write_test(iterations);
    read_test();
  }

  void write_test(size_t iterations)
  {
    size_t range = _size;
    uint8_t * data = (uint8_t *) p_base();
    cpu_time_t start = rdtsc();
    for(unsigned i=0;i<iterations;i++) {
      data[RAND() % range]  = i & 0xff;
      pmem_persist(data, 1);
    }
    cpu_time_t end = rdtsc();
    PLOG("Byte_8_write: %ld cycles per iteration", (end - start)/iterations);
  }

  void read_test()
  {
    uint8_t * data = (uint8_t *) p_base();
    size_t range = _size;

    { /* extra make sure to invalidate cache */
      uint8_t * p = (uint8_t *) malloc(MB(32));
      for(unsigned i=0;i<MB(32);i++) {
        p[i] = i;
      }
      free(p);
    }
    
    clflush_area(data, range);

    wmb();

    unsigned iterations = range / CACHE_LINE_SIZE;
    unsigned long x;

    cpu_time_t start = rdtsc();
    /* stride cache lines */
    for(unsigned i=0;i<iterations;i++) {
      x += data[i*CACHE_LINE_SIZE];
    }
    cpu_time_t end = rdtsc();
    PLOG("Byte_8_read: %ld cycles per iteration", (end - start)/iterations);
  }
    

private:
  size_t _size;
  
};


class Memset_tx_test : public Pmem_base
{
  static constexpr unsigned DATASIZE = MB(8);
public:
  Memset_tx_test(std::string filename,
                 size_t iterations = 100000)
    : Pmem_base(filename, MB(8)), _size(MB(8)) {
    block_write(iterations, 8);
    block_write(iterations, 64);
    block_write(iterations, 128);
  }

  void block_write(size_t iterations, size_t block_size)
  {
    std::vector<void*> index_list;
    size_t n_blocks = _size / block_size;
    
    uint8_t * ptr = (uint8_t *) p_base();
    for(unsigned i=0;i<n_blocks;i++) {
      index_list.push_back((void*)(ptr + (i*block_size)));
    }
    std::random_shuffle(index_list.begin(), index_list.end());
    
    
    cpu_time_t start = rdtsc();
    for(auto& p: index_list) {

      TX_BEGIN(_pop) {
        pmemobj_tx_add_range_direct(p,block_size);
        memset(p,0xf,block_size);
      }
      TX_END
    }
    
    cpu_time_t end = rdtsc();
    PLOG("Memset_tx (blocksize=%lu): %ld cycles per iteration",
         block_size, (end - start)/iterations);
  }

    

private:
  size_t _size;
  
};


/*
  Microbenchmarks:

  8-byte ATOMIC reads/writes
  4 and 8 byte bit sweeps
  write transactions (16,32,64,128)
  write zeroing (1,2,4,8,16,32,64,128,256,512)

 */

int main(int argc, char * argv[])
{
  set_cpu_affinity(0x2);
  //  Pmem_base pmem("data.pmem", MB(8));
  //  Byte_8_rw_test t1("data.pmem", MB(8));

  Memset_tx_test t2("data.pmem");
  
	return 0;
}
