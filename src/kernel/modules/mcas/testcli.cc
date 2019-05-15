#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <inttypes.h>
#include "mcas.h"

// prints LSB to MSB
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <vector>

#include <nupm/nupm.h>

#define REDUCE_KB(X) (X >> 10)
#define REDUCE_MB(X) (X >> 20)
#define REDUCE_GB(X) (X >> 30)
#define REDUCE_TB(X) (X >> 40)

#define KB(X) (X << 10)
#define MB(X) (X << 20)
#define GB(X) (((unsigned long) X) << 30)
#define TB(X) (((unsigned long) X) << 40)

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000 /* arch specific */
#endif

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif

#ifndef MAP_HUGE_MASK
#define MAP_HUGE_MASK 0x3f
#endif

int main()
{
  //  Devdax_manager ddm({{"/dev/dax0.3", 0x9000000000, 0},
  size_t size = 8000000;
  int fd = open("/dev/mcas", O_RDWR);
  assert(fd != -1);
  
  //  void * ptr = aligned_alloc(KB(4), MB(4));
  void * ptr;
  int flags = (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_FIXED);
  //  flags |= (shift & MAP_HUGE_MASK) << MAP_HUGE_SHIFT;
  ptr = mmap(((void*) 0x910000000),
             size,
             PROT_READ|PROT_WRITE,
             flags,
             -1,
             0);

  assert(ptr);
  memset(ptr, 0, MB(4));
  printf("allocated ptr=%p\n", ptr);
  

  IOCTL_EXPOSE_msg ioparam;
  ioparam.token = 666;
  ioparam.vaddr = ptr;
  ioparam.vaddr_size = size;

  int rc = ioctl(fd, IOCTL_CMD_EXPOSE, &ioparam);  //ioctl call
  if(rc != 0) {
    printf("ioctl failed: rc=%d\n", rc);
    close(fd);
    return 0;
  }

  munmap(ptr, size);
  close(fd);
}
