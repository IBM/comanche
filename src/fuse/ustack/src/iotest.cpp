#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <core/ipc.h>
#include <core/uipc.h>

#include "ustack_client.h"


int main()
{
  Ustack_client ustack("ipc:///tmp//ustack.ipc");

  ustack.get_uipc_channel();
  ustack.get_shared_memory(MB(4));
  ustack.get_io_memory(8);
  ustack.send_command();
  
  FILE * fp = fopen("./fs/fio.blob","w+");
  if(fp==NULL) {
    perror("error:");
  }

  //  char buf[256];
  //  size_t rc = fread(buf, 4096, 1, fp);
  
  #if 0
  
  for(unsigned i=0;i<100;i++) {
    size_t rc = fread(buf, 256, 1, fp);
    assert(rc == 256);
  }
  #endif
  
  fclose(fp);
  return 0;
}
