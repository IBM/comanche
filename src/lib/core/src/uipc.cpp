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
#include <common/utils.h>
#include <common/exceptions.h>

#include "uipc.h"

namespace Core {
namespace UIPC {

Shared_memory::Shared_memory(std::string name, size_t n_pages)
{
  _vaddr =  __negotiate_addr_create(name.c_str(),
                                    n_pages * PAGE_SIZE);
  assert(_vaddr);
  _size_in_pages = n_pages * PAGE_SIZE;
}

Shared_memory::Shared_memory(std::string name)
{
  _vaddr =  __negotiate_addr_connect(name.c_str(),&_size_in_pages);
  assert(_vaddr);
}


Shared_memory::~Shared_memory()
{
}

void * Shared_memory::get_addr()
{
}


static addr_t VADDR_BASE=0x8000000000;

void *
Shared_memory::
__negotiate_addr_create(const char * path_name,
                                      size_t size_in_bytes)
{
  /* create FIFO - used to negotiate memory address) */
  umask(0);

  std::string name = path_name;
  name += "-bootstrap";

  unlink(name.c_str());
  mkfifo(name.c_str(), 0666);

  FILE * fd_s2c = fopen(name.c_str(), "w");
  FILE * fd_c2s = fopen(name.c_str(), "r");
  assert(fd_c2s && fd_s2c);
  
  addr_t vaddr = VADDR_BASE;
  void * ptr = nullptr;
  
  do {
    ptr = mmap((void*) vaddr,
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
    
    char response;
    do {
      usleep(1000);
      response = fgetc(fd_c2s);
    }
    while(!response);

    if(response == 'Y') break;  /* 'Y' signals agreement */
    
    /* remove previous mapping and slide along */
    munmap(ptr, size_in_bytes);
    vaddr += size_in_bytes;
    
  } while(1);

  fclose(fd_c2s);
  fclose(fd_s2c);
  unlink(name.c_str());

  return ptr;
}

void *
Shared_memory::
__negotiate_addr_connect(const char * path_name,
                         size_t * size_in_bytes_out)
{
  umask(0);
    
  std::string name = path_name;
  name += "-bootstrap";

  FILE * fd_s2c = fopen(name.c_str(), "r");
  FILE * fd_c2s = fopen(name.c_str(), "w");
  assert(fd_c2s && fd_s2c);

  addr_t vaddr = 0;
  size_t size_in_bytes;
  void * ptr = nullptr;
  
  do {
    vaddr = 0;
    if(fread(&vaddr, sizeof(void*), 1, fd_s2c) != 1) {
      perror("");
      throw General_exception("fread failed (vaddr) in uipc_connect_shared_memory");
    }

    if(fread(&size_in_bytes, sizeof(size_t),1,fd_s2c) != 1) 
      throw General_exception("fread failed (size_in_bytes) in uipc_connect_shared_memory");

    ptr = mmap((void*) vaddr,
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
  
  fclose(fd_c2s);
  fclose(fd_s2c);

  *size_in_bytes_out = size_in_bytes;
  return ptr;
}

}} // Core::UIPC


/** 
 * 'C' exported functions
 * 
 */
extern "C" channel_t uipc_create_channel(const char * path_name,
                                         size_t message_size,
                                         size_t queue_size)
{
  return NULL;
}

extern "C" channel_t uipc_connect_channel(const char * path_name)
{
  return NULL;
}

extern "C" status_t uipc_close_channel(channel_t channel)
{
  return E_NOT_IMPL;
}

extern "C" void * uipc_alloc_message(channel_t channel)
{
  return NULL;
}

extern "C" status_t uipc_free_message(channel_t channel, void * message)
{
}

extern "C" status_t uipc_send(channel_t channel, void* data)
{
}

extern "C" status_t uipc_pop(channel_t channel, void*& data_out)
{
}

