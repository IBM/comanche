#include <stdio.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>

#include <common/logging.h>
#include <common/utils.h>
#include <common/cycles.h>

#define DO_MSYNC
#define CPU_MHZ 2400

void do_seq_write(void * addr, size_t n_elements, size_t size)
{
  int rc;
  char * p = (char*) addr;

  PLOG("Sequential write.");
  /* seq write */
  cpu_time_t begin = rdtsc();
  for(unsigned long i=0;i<n_elements;i++) {
    p[PAGE_SIZE*i + 2] = 0xf;
    //       syncfs(fd);
    if(i % 1000000 == 0) PLOG("page: %lu", i);
  }

  rc = munmap(addr,size); // causes flush
  assert(rc==0);

  cpu_time_t end = rdtsc();
  uint64_t usec = (end - begin) / CPU_MHZ;
  PINF("%lu usec/page fault", usec / n_elements);
}



void do_stride_write(void * addr, size_t n_elements, size_t size)
{
  int rc;
  char * p = (char*) addr;

  PLOG("Stride write.");

  /* seq write */

  const unsigned HOP=8;
  assert(n_elements % HOP == 0);
  unsigned long count = 0;

  rc = madvise(addr, size, MADV_RANDOM);
  assert(rc == 0);

  cpu_time_t begin = rdtsc();
  
  for(unsigned long h=0;h<HOP;h++) {
    for(unsigned long i=h;i<n_elements;i+=HOP) {
      p[PAGE_SIZE*i] = 0xf;
      if(++count % 100000 == 0) PLOG("count: %lu", count);
    }
  }

  rc = munmap(addr,size); // causes flush
  assert(rc==0);

  cpu_time_t end = rdtsc();
  uint64_t usec = (end - begin) / CPU_MHZ;
  PINF("%lu usec/page fault", usec / n_elements);
}


void do_random_write(void * addr, size_t n_elements, size_t size)
{
  int rc;
  char * p = (char*) addr;

  std::vector<addr_t> pages;

  PLOG("Creating random write sequence..");
  for(unsigned long i=0;i<n_elements;i++) {
    pages.push_back(i);
  }
  std::random_device rd;
  std::default_random_engine e(0);
  std::mt19937 g(e());
 
  std::shuffle(pages.begin(), pages.end(), g);

  rc = madvise(addr, size, MADV_RANDOM);
  assert(rc == 0);

  PLOG("Starting random write..");
  /* seq write */
  cpu_time_t begin = rdtsc();

  unsigned long count = 0;
  
  for(auto& i: pages) {
    p[PAGE_SIZE*i] = 0xf;

#ifdef DO_MSYNC
    msync(&p[PAGE_SIZE*i], PAGE_SIZE, MS_SYNC);
#endif
    //    PLOG("page: %ld",i);
    if(++count % 100000 == 0) PLOG("count: %lu", count);
  }

  rc = munmap(addr,size); // causes flush
  assert(rc==0);

  cpu_time_t end = rdtsc();
  uint64_t usec = (end - begin) / CPU_MHZ;
  PINF("%lu usec/page fault", usec / n_elements);
}



int main()
{
  int rc;
  int fd = open("/dev/nvme0n1",
                O_RDWR | O_DIRECT);
   
  assert(fd != -1);

  size_t n_elements = 80000000ULL; /* 305GB in 4K lbaf */
  //size_t n_elements = 40000000ULL; /* 152GB */
  //size_t n_elements = 8000000ULL; /* 30GB in 4K lbaf */
  size_t size = n_elements * KB(4);
  //  rc = fallocate(fd, 0, 0, size);
  //  assert(rc == 0); 

  void * addr = mmap(NULL,
                     size,
                     PROT_READ | PROT_WRITE,
                     MAP_SHARED,
                     fd, 0);
  assert(addr);
  PLOG("addr=%p",addr);


  //  do_seq_write(addr, n_elements, size);
  //do_stride_write(addr, n_elements, size);
  do_random_write(addr, n_elements, size);

  close(fd);
  PLOG("Done OK.:");
  return 0;
}

