#include <sys/mman.h>
#include <stdio.h>
#include <signal.h>
#include <errno.h>
#include <assert.h>
#include <set>
#include <common/utils.h>
#include <common/logging.h>
#include <common/cycles.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int fd = -1;

void *open_mapped_file(std::string filename, size_t size,
                       addr_t vaddr) {

  /* open file or create if it does not exist */
  filename = filename;
  fd = open(filename.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
  if (fd < 0) {
    if (errno == ENOENT) {
      fd = open(filename.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
      assert(fd >= 0);

      int rc = posix_fallocate(fd, 0, size);
      assert(rc == 0);

      PLOG("created %s (%ld bytes)", filename.c_str(), size);
    } else {
      PERR("errno=%d", errno);
      PERR("unexpected condition in open_mapped_file");
      return NULL;
    }
  } else {
    PLOG("opened existing file (%s)", filename.c_str());
  }

  /* map memory */
  void *ptr = ::mmap((void *)vaddr, size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_FIXED, fd, 0);
  assert(ptr != (void *)-1);

  //close(fd); // no need to keep open once mapped
  PLOG("mapped at memory %p", ptr);
  return ptr;
}

#define SIZE MB(128)
//#define FILENAME "/dev/hugepages-1G/nvme_memorymap_0"
#define FILENAME "/dev/hugepages/nvme_memorymap_11"
//#define FILENAME "./foobar.mmap"

int main(int argc, char * argv[])
{

  if(argc > 1) {

    int pid;
    if((pid = fork()) == 0) {
      
      PLOG("writing...");
      /* simple 1G write */
      char * p = (char *) open_mapped_file(FILENAME,
                                           SIZE,0x900000000);
      
      size_t s = 0;
      while(s < SIZE) {
        p[s] = 'a' + (s % 24);
        //    memset(p,'a',);
        s+=512;
      }
      ::msync(p,SIZE, MS_SYNC);
      
    }
    else {
      sleep(1);      
      int fd = open(FILENAME, O_RDWR, S_IRUSR | S_IWUSR);

      void *ptr = ::mmap(NULL, MB(2), PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, MB(2)); // offset must be according to hugepage size.
      assert(ptr != (void*) -1);
      PLOG("child mapping OK. ptr=%p", ptr);
      close(fd);
    }
  }
  else {
    PLOG("checking...");
    char * p = (char *) open_mapped_file(argv[1],GB(1),0x900000000);

    size_t s = 0;
    while(s < SIZE) {
      if(p[s] != ('a' + (s % 24))) {
        PERR("data integrity failure");
      }
      //    memset(p,'a',);
      s+=512;
    }
  }
  return 0;
  
}
