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
#if 0
  void * ptr = malloc(4096);
  memset(ptr,'a',4096);
  int fd = open("foobar.dat", O_SYNC | O_CREAT | O_TRUNC, O_WRONLY);
  assert(fd != -1);
  for(unsigned i=0;i<0;i++) {
    ssize_t res = write(fd, ptr, 4096);
  }
  close(fd);
  PLOG("done!");
#endif
  
#if 1
  //Ustack_client ustack("ipc:///tmp//ustack.ipc", 64);
  Ustack_client ustack("ipc:///tmp//kv-ustack.ipc", 64);

  PINF("[test]: ustack client constructed");

  ustack.get_uipc_channel();
  //ustack.get_shared_memory(MB(4));

  PINF("[test]: got channel and shared memory");

  //ustack.send_command();

 
  void  * data;
  constexpr size_t data_sz = 4096;

  data = ustack.malloc(data_sz);
  assert(data);

  int fd = ustack.open("./mymount/test.dat",O_CREAT|O_RDWR, 0666);
  assert(fd >0);
  PINF("[test]: file opened");

  // shall I use write instead of fwrite?
  ustack.write(fd, data, data_sz);
  //  size_t rc = fread(buf, 4096, 1, fp);
  
  #if 0
  char buf[256];
  for(unsigned i=0;i<100;i++) {
    size_t rc = fread(buf, 256, 1, fp);
    assert(rc == 256);
  }
  #endif
  
  ustack.close(fd);
  PINF("[test]: file closed");

#endif
  return 0;
}
