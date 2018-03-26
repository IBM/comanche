#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <core/ipc.h>
#include <core/uipc.h>
#include <common/cpu.h>

#include "ustack_client.h"


int main()
{
  void * ptr = malloc(4096);
  memset(ptr,'a',4096);
  int fd = open("foobar.dat", O_SYNC | O_CREAT | O_TRUNC | O_WRONLY);
  assert(fd != -1);
  for(unsigned i=0;i<0;i++) {
    ssize_t res = write(fd, ptr, 4096);
  }
  close(fd);
  PLOG("done!");
  
  #if 0
  Ustack_client ustack("ipc:///tmp//ustack.ipc", 64);

  ustack.get_uipc_channel();
  ustack.get_shared_memory(MB(4));
  ustack.send_command();

  FILE * fp = fopen("./fs/fio.blob","w+");
  if(fp==NULL) {
    perror("error:");
  }

  //  size_t rc = fread(buf, 4096, 1, fp);
  
  #if 0
  char buf[256];
  for(unsigned i=0;i<100;i++) {
    size_t rc = fread(buf, 256, 1, fp);
    assert(rc == 256);
  }
  #endif
  
  fclose(fp);

  #endif
  return 0;
}
