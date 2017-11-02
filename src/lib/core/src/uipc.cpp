/*
  Copyright [2017] [IBM Corporation]

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>
#include <string>
#include <mutex>
#include <common/exceptions.h>

#include "uipc.h"

static addr_t VADDR_BASE=0x8000000000;

static constexpr bool option_DEBUG = true;

extern "C" void * uipc_create_shared_memory(const char * path_name,
                                            size_t size_in_bytes,
                                            int* fd_out)
{
  /* create FIFO - used to negotiate memory address) */
  umask(0);

  std::string name = path_name;
  name += "-bootstrap";

  mkfifo(name.c_str(), 0666);

  FILE * fd_s2c = fopen(name.c_str(), "w");
  FILE * fd_c2s = fopen(name.c_str(), "r");
  assert(fd_c2s && fd_s2c);
  
  addr_t vaddr = VADDR_BASE;
  do {
    void * ptr = mmap((void*) vaddr,
                      size_in_bytes,
                      PROT_NONE,
                      MAP_SHARED | MAP_ANONYMOUS | MAP_FIXED,
                      0, 0);
    if(ptr == (void*) -1) {
      vaddr += size_in_bytes;
      continue;
    }
               
    /* send proposal */
    if(option_DEBUG)
      PLOG("sending vaddr proposal: %p", ptr);

    if(fwrite(&ptr, sizeof(void*), 1, fd_s2c) != 1)
      throw General_exception("write failed in uipc_accept_shared_memory (ptr)");

    if(fwrite(&size_in_bytes, sizeof(size_t), 1, fd_s2c) != 1)
      throw General_exception("write failed in uipc_accept_shared_memory (size)");

    fflush(fd_s2c);
    
    if(option_DEBUG)
      PLOG("waiting for response..");

    char response = fgetc(fd_c2s);
    assert(response);
    if(response == 'Y') break;  /* 'Y' signals agreement */
    
    /* remove previous mapping and slide along */
    munmap(ptr, size_in_bytes);
    vaddr += size_in_bytes;
    
  } while(1);

  PLOG("acceptor agreed on shared memory %lx", vaddr);
  fclose(fd_c2s);
  fclose(fd_s2c);
  unlink(name.c_str());
    
  return nullptr;
}
  
extern "C" void * uipc_connect_shared_memory(const char * path_name, int* fd_out)
{
  umask(0);
  
  std::string name = path_name;
  name += "-bootstrap";

  FILE * fd_s2c = fopen(name.c_str(), "r");
  FILE * fd_c2s = fopen(name.c_str(), "w");
  assert(fd_c2s && fd_s2c);

  addr_t vaddr = 0;
  size_t size_in_bytes;
  do {
    vaddr = 0;
    if(fread(&vaddr, sizeof(void*), 1, fd_s2c) != 1) {
      perror("");
      throw General_exception("fread failed (vaddr) in uipc_connect_shared_memory");
    }

    if(fread(&size_in_bytes, sizeof(size_t),1,fd_s2c) != 1) 
      throw General_exception("fread failed (size_in_bytes) in uipc_connect_shared_memory");

    void * ptr = mmap((void*) vaddr,
                      size_in_bytes,
                      PROT_NONE,
                      MAP_SHARED | MAP_ANON | MAP_FIXED,
                      0, 0);

    char answer;
    if(ptr == (void*)-1) {
      answer = 'N';
      if(fwrite(&answer, sizeof(answer), 1, fd_c2s) != 1)
        throw General_exception("write failed");
      fflush(fd_c2s);
    }
    else {
      answer = 'Y';
      if(fwrite(&answer, sizeof(answer),1,fd_c2s) != 1)
        throw General_exception("write failed");
      fflush(fd_c2s);
      break;
    }    
  }
  while(1);
  
  PLOG("agreed on shared memory %lx", vaddr);
  fclose(fd_c2s);
  fclose(fd_s2c);
  unlink(name.c_str());
  
  return nullptr;
}
  
extern "C" void uipc_close_shared_memory(int fd)
{
  return;
}


