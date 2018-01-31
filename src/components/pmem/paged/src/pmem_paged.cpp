#include <sys/types.h>
#include <stdio.h>
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
#include <mutex>
#include <map>
#include <common/exceptions.h>
#include "pmem_paged.h"

using namespace Component;

/* because page faults need to be multiplexed onto multiple component
   instances */
static std::vector<Pmem_paged_component*> __global_inst_v;
static Common::Spin_lock                  __global_inst_v_lock;

struct pmem_handle_t
{
  pmem_handle_t(void* vaddr_p, size_t size_p)  :
    vaddr(vaddr_p), size(size_p) {
    assert(vaddr_p != nullptr);
    assert(size_p > 0);
  }
  
  void * vaddr;
  size_t size;
};

static void * handle_to_vaddr(void* handle)
{
  return reinterpret_cast<void*>(reinterpret_cast<pmem_handle_t*>(handle)->vaddr);
}

static size_t handle_to_size(void* handle)
{
  return reinterpret_cast<pmem_handle_t*>(handle)->size;
}

/** 
 * Trampoline which connects static callback to instances of the paging slab
 * 
 * @param addr 
 * @param fec 
 * @param tf 
 */
static void __segv_handler_trampoline(int sig, siginfo_t* si, void* context)
{
  for (auto& e : __global_inst_v) {
    if (e->pf_handler((addr_t)(si->si_addr))) {
      return;
    }
    else {
      panic("custom pf handler failed!!.");
    }
  }

  // __default_segv_handler(addr,fec,tf);
}

__attribute__((constructor)) static void __ctor()
{
  /* attach SEGV handler */
  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = __segv_handler_trampoline;
  if (sigaction(SIGSEGV, &sa, NULL) == -1) assert(0);
}


Pmem_paged_component::
Pmem_paged_component(std::string owner_id, IPager* pager, bool force_init)
  : _pager(pager), _owner_id(owner_id)
{
  if(!_pager) throw API_exception("%s: IPager interface param invalid", __PRETTY_FUNCTION__);
  _pager->add_ref();

  __global_inst_v_lock.lock();
  __global_inst_v.push_back(this);
  __global_inst_v_lock.unlock();

  _fd_xms = ::open("/dev/xms", O_RDWR, 0666);
  if(_fd_xms == -1) throw Constructor_exception("unable to open XMS module");
  PLOG("XMS module open OK.");
}

Pmem_paged_component::
~Pmem_paged_component()
{

  assert(_pager);
  _pager->release_ref();
  ::close(_fd_xms);
}


bool Pmem_paged_component::pf_handler(addr_t fault_addr)
{
  //  PLOG("pf_handle addr=%lx", fault_addr);
  _fault_count++;
  
  addr_t new_phys = 0;
  addr_t evict_vaddr = 0;

  /* request resolution from pager component */
  addr_t page = fault_addr & ~(0xFFFUL);
  _pager->request_page(page, &new_phys, &evict_vaddr);
  assert(new_phys);

  if(option_DEBUG)
    PLOG("pager result: fault=0x%lx new=0x%lx evict=0x%lx", fault_addr, new_phys, evict_vaddr);


  /* evict page (if necessary) */
  static unsigned evict_count = 0;
  if(evict_vaddr) {

    int rc;
    // rc = munmap((void*)evict_vaddr, PAGE_SIZE);
    // if(rc == -1) throw General_exception("munmap failed");

    assert((evict_vaddr & 0xFFFUL) == 0);
    rc = mprotect((void*)evict_vaddr, PAGE_SIZE, PROT_NONE);
    if(rc == -1)
      throw General_exception("%s: mprotect failed (%p) rc=%d",
                              __PRETTY_FUNCTION__, (void*)evict_vaddr, errno);
  }
  /* map new page */
  void  * ptr = ::mmap((void*) page,
                       PAGE_SIZE,
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_FIXED,
                       _fd_xms,
                       (off_t) new_phys);  // offset in the file is physical address

  if (ptr == (void *)-1 || ptr != ((void*)page))
    throw General_exception("%s: mmap failed (%d)", __PRETTY_FUNCTION__, errno);

  if(option_DEBUG)
    PLOG("Pmem-paged: mapped %lx to %p", new_phys, ptr);
  
  return true;
}

static std::mutex _size_map_lock;
static std::map<void*,size_t> _size_map;

IPersistent_memory::pmem_t
Pmem_paged_component::
open(std::string id, size_t size, int numa_node, bool& reused, void*& vptr)
{
  assert(_pager);

  size = round_up_page(size);

  void * addr = _pager->get_region(id, size, reused);

  /* allocate virtual memory only */
  void * maddr = mmap(addr,
                      size,
                      PROT_NONE,
                      MAP_NORESERVE | MAP_FIXED | MAP_SHARED | MAP_ANONYMOUS,
                      -1, 0);
  
  if (maddr != addr || maddr == MAP_FAILED)
    throw General_exception("%s: mmap failed in allocate:%d addr=%p",
                            __PRETTY_FUNCTION__, errno, addr);

  if(option_DEBUG)
    PLOG("Address returned by mmap() = %p", addr);

  vptr = addr;

  /* record size of allocation */
  {
    std::lock_guard<std::mutex> g(_size_map_lock);
    _size_map[addr] = size;
  }
  return static_cast<void*>(new pmem_handle_t(addr,size));
}

/** 
 * Free a previously allocated persistent memory region
 * 
 * @param handle Handle to persistent memory region
 */
void
Pmem_paged_component::
close(IPersistent_memory::pmem_t handle, int flags)
{
  if(flags > 0)
    throw API_exception("%s: flag not supported", __PRETTY_FUNCTION__);
  
  void * ptr = handle_to_vaddr(handle);  
  if(!ptr)
    throw API_exception("pmem_page_component: bad handle");
  
  std::lock_guard<std::mutex> g(_size_map_lock);
  size_t msize = _size_map[ptr];

  PLOG("msize=%lu handle2size=%lu", msize, handle_to_size(handle));
  assert(msize == handle_to_size(handle));
  
  int rc = munmap(ptr,msize);
  if(rc)
    throw General_exception("%s: munmap failed", __PRETTY_FUNCTION__);
  _size_map.erase(ptr);

  _pager->clear_mappings((addr_t)ptr, msize);
  PLOG("pmem freed (%p)", ptr);
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
  addr_t vaddr = reinterpret_cast<addr_t>(p);
  std::lock_guard<std::mutex> g(_size_map_lock);
  for(auto& r: _size_map) {
    if((vaddr >= (addr_t)r.first) && (vaddr <= ((addr_t)r.first + r.second)))
      return true;
  }
  return false;
}

/** 
 * Flush all volatile data to peristent memory in a non-transacation context.
 * 
 */
void
Pmem_paged_component::
persist(pmem_t handle)
{
  _pager->flush(reinterpret_cast<addr_t>(handle_to_vaddr(handle)),
                handle_to_size(handle));
}

void
Pmem_paged_component::
persist_scoped(pmem_t handle, void *ptr, size_t size)  
{
  _pager->flush(reinterpret_cast<addr_t>(ptr), size);
}

size_t
Pmem_paged_component::
get_size(pmem_t handle)
{
  void * ptr = reinterpret_cast<void*>(handle);
  std::lock_guard<std::mutex> g(_size_map_lock);
  return _size_map[ptr];
}

/** 
 * Start transaction
 * 
 */
void
Pmem_paged_component::
tx_begin(pmem_t handle)
{
  throw API_exception("%s: not supported", __FUNCTION__);
}

/** 
 * Commit transaction
 * 
 * 
 * @return S_OK or E_FAIL
 */
status_t
Pmem_paged_component::
tx_commit(pmem_t handle)
{
  throw API_exception("%s: not supported", __FUNCTION__);
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

