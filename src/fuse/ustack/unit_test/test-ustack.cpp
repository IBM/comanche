/*
 * Description:
 * ./exe to start the server, ./exe 1 to start client
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <core/ipc.h>
#include <core/uipc.h>
#include <common/cpu.h>
#include <core/physical_memory.h>
#include <sys/wait.h>

#include "ustack.h"
#include "ustack_client.h"
#include "core/dpdk.h"

int main(int argc, char * argv[])
{
  bool is_server = true;
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
  if(argc == 2) is_server= false;
  
#if 1

  int nr_iterations=100;

  if (is_server) {
		DPDK::eal_init(1024);
		PLOG("fuse_ustack: DPDK init OK.");


		Ustack *ustack;
    PMAJOR("[server]: tried to start");
		ustack = new Ustack("ipc:///tmp//kv-ustack.ipc");
    PMAJOR("[server]: started, press Enter to exit");
    getchar();

  }
  else{
    sleep(5);
    PMAJOR("[client]: tried to start");
    Ustack_client ustack("ipc:///tmp//kv-ustack.ipc", 64);
    PMAJOR("[client]: started");

    ustack.get_uipc_channel();
    ustack.get_shared_memory(MB(4));
    for(int i = 0; i < nr_iterations; i++){
      ustack.send_command();
    }

  }

#endif
  return 0;
}


