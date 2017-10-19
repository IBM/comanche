#include <sys/stat.h>
#include <sys/ioctl.h>
#include <poll.h>
#include <fcntl.h>
#include <unistd.h>
#include <common/logging.h>
#include <assert.h>
#include <stdio.h>
#include <thread>
#include <sys/mman.h>
#include <signal.h>

#include "../copager_msg.h"

bool b_exit = false;

void pager_thread()
{
  int fd = open("/dev/copager",O_RDWR);
  PLOG("/dev/coppager opened OK.");

#if 0
  struct pollfd pfd;
  pfd.fd = fd;
  pfd.events = 1;
  pfd.revents = 1;

  int rc = poll(&pfd, 1, 1000 /*timeout in ms*/);
  PLOG("poll returned: %d",rc);
#endif

  service_request_t sr;
  __builtin_memset(&sr, 0, sizeof(service_request_t));

  while(!b_exit) {    
    int rc = ioctl(fd, COPAGER_IOCTL_TAG_SERVICE, &sr);  
    PLOG("ioctl got sr: %d (%lx,%p,%d)", rc, sr.addr[0], sr.signal,sr.pid);

    /* set up result */
    sr.addr[ADDR_IDX_PHYS] = 0x13524e000; /* test physical address */
    sr.addr[ADDR_IDX_INVAL] = 0x0;
  }
  
  close(fd);
}

int main()
{
  std::thread pfh_thread(pager_thread);

  int fd = open("/dev/copager",O_RDWR);
  assert(fd != -1);

  size_t sz = 4096;
  char * ptr = (char *) mmap(NULL,
                             sz,//the last page is for communcation
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED,
                             fd, 0);


  PLOG("About to fault ...");
  ptr[0] = 0xf;
  
  PLOG("Clean up...");
  munmap(ptr, sz);

  close(fd);
  b_exit = true;
  pthread_kill(pfh_thread.native_handle(),SIGUSR1); /* todo set SIGUSR1 handler */

  pfh_thread.join();
}
  
