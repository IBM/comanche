/*
   Copyright [2017-2019] [IBM Corporation]
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
#include <sys/types.h>
#include <stdio.h>
#include <linux/userfaultfd.h>
#include <pthread.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <signal.h>
#include <poll.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>

#include <common/exceptions.h>
#include "pmem_paged_component.h"
#include "copager_msg.h"

using namespace Component;

Pmem_paged_component::
Pmem_paged_component(std::string heap_set_id, IPager* pager) : _pager(pager)
{
  if(!_pager) throw API_exception("%s: IPager interface param invalid", __PRETTY_FUNCTION__);
  _pager->add_ref();
  
  /* open kernel module */
  _fdmod = open("/dev/copager",O_RDWR);

  /* start pager thread */
  start_thread();

}

Pmem_paged_component::
~Pmem_paged_component()
{
  //  assert(_pf_thread);
  _pf_thread_status = THREAD_EXIT;
  //  _pf_thread->join();
  //  delete _pf_thread;

  close(_fdmod);

  assert(_pager);
  _pager->release_ref();
}

status_t
Pmem_paged_component::start_thread()
{
  _pf_thread_status = 0;
  _pf_thread = new std::thread(&Pmem_paged_component::pf_thread_entry, this);
  assert(_pf_thread);
  while(_pf_thread_status == THREAD_UNINIT) usleep(100);

  return S_OK;
}


void
Pmem_paged_component::
pf_thread_entry()
{
  using namespace Component;
  assert(_pager);
  
  service_request_t sr;
  __builtin_memset(&sr, 0, sizeof(service_request_t));

  /* open kernel module */
  int fd_copager = open("/dev/copager",O_RDWR);
  if(fd_copager == -1) throw General_exception("unable to open fd_copager");

  addr_t last_vaddr = 0;

  _pf_thread_status = THREAD_RUNNING;

  while(!_exit_thread) {    
    int rc = ioctl(fd_copager, COPAGER_IOCTL_TAG_SERVICE, &sr);  

    if(rc == -1) {
      PLOG("[pf_thread]: timed out, no requests");
      continue;
    }

    if(option_DEBUG)
      PLOG("[pf_thread]: USER-LEVEL service request: %d (%lx,%p,%d)",
           rc, sr.addr[ADDR_IDX_FAULT], sr.signal, sr.pid);

    assert(_pager);

    // if(!_pager->check_valid(sr.addr[ADDR_IDX_FAULT])) {
    //   throw General_exception("unexpected fault at %p", sr.addr[ADDR_IDX_FAULT]);
    // }
    /* resolve page fault through pager; action is taken when
       we reenter the ioctl */
    _pager->request_page(sr.addr[ADDR_IDX_FAULT],&sr.addr[ADDR_IDX_PHYS],&sr.addr[ADDR_IDX_INVAL]);
    
    // /* set up result */
    // sr.addr[ADDR_IDX_PHYS] = iob_phys; /* test physical address */
    // sr.addr[ADDR_IDX_INVAL] = last_vaddr;
    // last_vaddr = tmp; /* one in, one out */
  }

  
  close(fd_copager);
}



/** 
 * Create a region of persistent memory
 * 
 * @param size Size of region in bytes
 * @param vptr [out] Pointer to region
 * 
 * @return Handle to persistent memory region
 */
IPersistent_memory::pmem_t
Pmem_paged_component::
allocate(std::string id, size_t size, void** vptr)
{
  assert(_fdmod);
  assert(_pager);

  void * addr = _pager->get_region(id, size);
  /* allocate virtual memory only */
  void * maddr = mmap(addr,
                      size,
                      PROT_READ | PROT_WRITE,
                      MAP_FIXED | MAP_SHARED,
                      _fdmod, 0);
  
  if (maddr != addr || maddr == MAP_FAILED)
    throw General_exception("mmap failed in allocate:%d", errno);

  PLOG("Address returned by mmap() = %p", addr);

  *vptr = addr;
  //  _pager->add_region(addr, size);


  
  return addr;
}

/** 
 * Free a previously allocated persistent memory region
 * 
 * @param handle Handle to persistent memory region
 */
void
Pmem_paged_component::
free(IPersistent_memory::pmem_t handle)
{
}

/** 
 * Determine if a virtual address is in persistent memory
 * 
 * @param p Virtual address pointer
 * 
 * @return True if belonging to persistent memory
 */
bool
Pmem_paged_component::
is_pmem(void * p)
{
}

/** 
 * Flush all volatile data to peristent memory in a non-transacation context.
 * 
 */
void
Pmem_paged_component::
persist()
{
}

/** 
 * Start transaction
 * 
 */
void
Pmem_paged_component::
tx_begin()
{
}

/** 
 * Commit transaction
 * 
 * 
 * @return S_OK or E_FAIL
 */
status_t
Pmem_paged_component::
tx_commit()
{
}



/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Pmem_paged_component_factory::component_id()) {
    return static_cast<void*>(new Pmem_paged_component_factory());
  }
  else
    return NULL;
}

