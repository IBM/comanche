#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <string>
#include <set>
#include <libpmemobj.h>
#include <libpmem.h>

#include <common/logging.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <common/cycles.h>
#include <common/rand.h>

#include "pmem_helper.h"

//#define LAYOUT_NAME "intro_0" /* will use this in create and open */
#define MAX_BUF_LEN 10 /* maximum length of our buffer */

#define RAND genrand64_int64


class Atomic_8_rw_test : public Pmem_base
{
  static constexpr unsigned DATASIZE = MB(8);
public:
  Atomic_8_rw_test(std::string filename, size_t iterations) : Pmem_base(filename, MB(8)), _size(MB(8)) {
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
    PLOG("Atomic_8_write: %ld cycles per iteration", (end - start)/iterations);
  }

  void read_test()
  {
    uint8_t * data = (uint8_t *) p_base();
    size_t range = _size;
    clflush_area(data, range);

    wmb();

    unsigned iterations = range / CACHE_LINE_SIZE;
    unsigned long x;

    cpu_time_t start = rdtsc();
    /* stride cache lines */
    for(unsigned i=0;i<iterations;i++) {
      x += data[i*CACHE_LINE_SIZE];
      asm("nop");
    }
    cpu_time_t end = rdtsc();
    PLOG("Atomic_8_read: %ld cycles per iteration", (end - start)/iterations);
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
  //  Pmem_base pmem("data.pmem", MB(8));
  Atomic_8_rw_test t1("data.pmem", MB(8));
  
	return 0;
}
