#include <sys/types.h> //open
#include <fcntl.h> // O_flags
#include <unistd.h> // close
#include <common/logging.h>
#include <string.h>
#include <cstdlib>
/** Preload the ustack libray and those operations will be intecepted
 * 1. malloc free
 * 2. open/read/write*/
static const char * filepath = "foobar.dat";
int main(){
  int fd = open(filepath, 
      O_WRONLY |  O_CREAT,
      S_IRWXU);
  if(fd == -1){
    PERR("Open failed, delete file (%s)first", filepath);
    return -1;
  }
  void * ptr = malloc(4096);
  memset(ptr,'a',4096);

#if 0
  int fd = open("foobar.dat", O_SYNC | O_CREAT | O_TRUNC, O_WRONLY);
  assert(fd != -1);

#endif
  for(unsigned i=0;i<1;i++) {
    ssize_t res = write(fd, ptr, 4096);
  }

  close(fd);
  PLOG("done!");
  return 0;
}

