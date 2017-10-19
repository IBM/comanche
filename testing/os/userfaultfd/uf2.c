#define _GNU_SOURCE
#include <assert.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include <signal.h>
#include <poll.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <linux/userfaultfd.h>
#include <common/cycles.h>

#ifdef __NR_userfaultfd

int uffd;
cpu_time_t start_stamp;


static void *uffd_poll_thread(void *arg)
{
  unsigned long cpu = (unsigned long) arg;
  struct pollfd pollfd[2];
  struct uffd_msg msg;
  int ret;
  unsigned long offset;
  char tmp_chr;
  unsigned long userfaults = 0;

  pollfd[0].fd = uffd;
  pollfd[0].events = POLLIN;

  for (;;) {
    ret = poll(pollfd, 1, -1);
    if (!ret)
      fprintf(stderr, "poll error %d\n", ret), exit(1);
    if (ret < 0)
      perror("poll"), exit(1);
    if (!(pollfd[0].revents & POLLIN))
      fprintf(stderr, "pollfd[0].revents %d\n",
              pollfd[0].revents), exit(1);
    ret = read(uffd, &msg, sizeof(msg));
    if (ret < 0) {
      if (errno == EAGAIN)
        continue;
      perror("nonblocking read error"), exit(1);
    }
    if (msg.event != UFFD_EVENT_PAGEFAULT)
      fprintf(stderr, "unexpected msg event %u\n",
              msg.event), exit(1);
    printf("latency cycles: %ld\n", rdtsc() - start_stamp);
    printf("Page fault!!\n");
    exit(0);
    
    /* if (msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WRITE) */
    /*   fprintf(stderr, "unexpected write fault\n"), exit(1); */
    /* offset = (char *)(unsigned long)msg.arg.pagefault.address - area_dst; */
    /* offset &= ~(page_size-1); */
    /* if (copy_page(offset)) */
    /*   userfaults++; */
  }
  return (void *)userfaults;
}

int main(int argc, char * argv[])
{
  void *area;
  char *tmp_area;
  unsigned long nr;
  struct uffdio_register uffdio_register;
  struct uffdio_api uffdio_api;
  unsigned long cpu;
  int uffd_flags, err;
  //  unsigned long userfaults[nr_cpus];

  uffd = syscall(__NR_userfaultfd, O_CLOEXEC | O_NONBLOCK);

  if (uffd < 0) {
    fprintf(stderr,"userfaultfd syscall not available in this kernel\n");
    return 1;
  }
  
  uffd_flags = fcntl(uffd, F_GETFD, NULL);

  uffdio_api.api = UFFD_API;
  uffdio_api.features = 0;
  if (ioctl(uffd, UFFDIO_API, &uffdio_api)) {
    fprintf(stderr, "Error: UFFDIO_API\n");
    return 1;
  }

  if (uffdio_api.api != UFFD_API) {
    fprintf(stderr, "UFFDIO_API error %Lu\n", uffdio_api.api);
    return 1;
  }

  printf("uffd features:%llx\n",uffdio_api.features);

  size_t PAGE_SIZE = 4096;
  size_t nr_pages = 8;
  size_t mem_size = PAGE_SIZE * nr_pages;
  void * mem = aligned_alloc(PAGE_SIZE, mem_size);
  assert(mem);
  
  uffdio_register.range.start = (unsigned long) mem;
  uffdio_register.range.len = nr_pages * PAGE_SIZE;
  uffdio_register.mode = UFFDIO_REGISTER_MODE_MISSING;
  if (ioctl(uffd, UFFDIO_REGISTER, &uffdio_register)) {
    fprintf(stderr, "register failure\n");
    return 1;
  }

  unsigned long expected_ioctls = (1 << _UFFDIO_WAKE) |
    (1 << _UFFDIO_COPY) |
    (1 << _UFFDIO_ZEROPAGE);
  if ((uffdio_register.ioctls & expected_ioctls) !=
      expected_ioctls) {
    fprintf(stderr,
            "unexpected missing ioctl for anon memory\n");
    return 1;
  }

  pthread_t poll_thread;
  pthread_create(&poll_thread, NULL,
                 uffd_poll_thread, (void *)NULL);

  sleep(3);
  start_stamp = rdtsc();
  
  /* fault */
  memset(mem,0,mem_size);

  free(mem);
  return 0;
}

#else
#error userfaultfd system call not available
#endif
