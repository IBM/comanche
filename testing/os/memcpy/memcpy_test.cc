#include <assert.h>
#include <common/utils.h>
#include <common/cycles.h>
#include <common/logging.h>

#define CLOCK_MHZ 2400
#define SLAB_SIZE GB(1)
void simple_memcpy_test()
{
  void * p = malloc(SLAB_SIZE);
  void * q = malloc(SLAB_SIZE);

  assert(p);
  PLOG("Zeroing...");
  cpu_time_t start = rdtsc();
  memset(p,0,SLAB_SIZE);
  cpu_time_t end = rdtsc();
  PLOG("Zero'ed in %ld usec.",(end-start)/CLOCK_MHZ);

  PLOG("Zeroing (after paged)...");
  start = rdtsc();
  memset(p,0,SLAB_SIZE);
  end = rdtsc();
  PLOG("Zero'ed in %ld usec.",(end-start)/CLOCK_MHZ);

  memset(q,0,SLAB_SIZE);

  PLOG("Copying ...");
  start = rdtsc();
  memcpy(p,q,SLAB_SIZE);
  end = rdtsc();
  unsigned long usec = (end-start)/CLOCK_MHZ;
  float throughput = (1000000.0/((float)usec))/((float)(SLAB_SIZE/GB(1)));
  PLOG("Copied in %lu usec at %f GB/s",usec, throughput);
  free(p);
}
  

int main()
{
  simple_memcpy_test();
  return 0;
}
