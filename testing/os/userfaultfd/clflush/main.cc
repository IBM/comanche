#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>

#define CACHE_LINE_SIZE 64
#define wmb() asm volatile("sfence" ::: "memory")

inline uint64_t rdtsc()
{
  unsigned a, d;
  asm volatile("lfence");  // should be mfence for AMD
  asm volatile("rdtsc" : "=a"(a), "=d"(d));
  return ((unsigned long long)a) | (((unsigned long long)d) << 32);
}

inline void clflush_area(void* p, size_t size)
{
  //  assert(check_aligned(p,CACHE_LINE_SIZE));

  ssize_t ssize = size;
  //  PLOG("clflush_area: p=%p size=%ld", p, size);

  char* ptr = static_cast<char*>(p);

  while (ssize > 0) {
    __builtin_ia32_clflush((const void*)ptr);
    ssize -= CACHE_LINE_SIZE;
    ptr += CACHE_LINE_SIZE;
  }
  wmb();  // sfence
}

int main(int argc, char* argv[])
{
  if(argc==1) {
    printf("flushtest <num 4K pages>\n");
    return 0;
  }
  size_t num_pages = atoi(argv[1]);
  printf("allocating %ld pages.\n", num_pages);

  void * p = aligned_alloc(4096, 4096 * num_pages);
  assert(p);
  /* time cache flush */
  auto ts_begin = rdtsc();

  clflush_area(p,4096*num_pages);
  
  printf("took: %f cycles per 4K flush\n", ((float)(rdtsc() - ts_begin)) / (float) num_pages);
  free(p);

  return 0;
}
             
  
  
