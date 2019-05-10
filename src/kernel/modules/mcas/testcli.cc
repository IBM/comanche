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

#define REDUCE_KB(X) (X >> 10)
#define REDUCE_MB(X) (X >> 20)
#define REDUCE_GB(X) (X >> 30)
#define REDUCE_TB(X) (X >> 40)

#define KB(X) (X << 10)
#define MB(X) (X << 20)
#define GB(X) (((unsigned long) X) << 30)
#define TB(X) (((unsigned long) X) << 40)


int main()
{
  
  int fd = open("/dev/mcas", O_RDWR);
  assert(fd != -1);
  
  void * ptr = aligned_alloc(KB(4), MB(4));

  assert(ptr);
  memset(ptr, 0, MB(4));
  printf("allocated ptr=%p\n", ptr);
  

  IOCTL_EXPOSE_msg ioparam;
  ioparam.auth_token == 666;
  ioparam.vaddr = ptr;
  ioparam.vaddr_size = KB(4);

  int rc = ioctl(fd, IOCTL_CMD_EXPOSE, &ioparam);  //ioctl call
  if(rc != 0) {
    printf("ioctl failed: rc=%d\n", rc);
    close(fd);
    return 0;
  }

  close(fd);
}
