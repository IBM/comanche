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
  PINF("[test]: got channel and shared memory");

  char * data;
  size_t data_sz = 4096; 
  data = (char *)ustack.malloc(data_sz);

  strcpy(data, "helloworld, this is written using ustack writes");
  assert(data);

  int fd = ustack.open("./mymount/test.dat",O_CREAT|O_RDWR, 0666);
  //int fd = ustack.open("./regular.dat",O_CREAT|O_RDWR, 0666);
  assert(fd >0);
  PINF("[test]: file opened write");

  // shall I use write instead of fwrite?
  ustack.write(fd, data, data_sz);
  //  size_t rc = fread(buf, 4096, 1, fp);
  
  ustack.close(fd);
  ustack.free(data);
  PINF("[test]: file closed");

  /*
   * reopen the file and verify the results read
   */
  fd = ustack.open("./mymount/test.dat",O_RDONLY);
  assert(fd >0);
  PINF("[test]: file opened for read");

  char *data2 = (char *)ustack.malloc(data_sz);

  ustack.read(fd, data2, data_sz);
  PINF("file content read:\n\t%s", data2);

  ustack.close(fd);
  PINF("[test]: file closed");
  ustack.free(data2);

#endif
  return 0;
}
