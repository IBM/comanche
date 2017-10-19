/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#include <sys/types.h>
#include <stdio.h>
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
#include <mutex>
#include <map>
#include <common/exceptions.h>
#include "pmem_fixed.h"

using namespace Component;


/** 
 * Map vaddr to a block device
 * 
 * @param rm 
 * @param owner 
 * @param region_set_id 
 */
class Range_tracker
{
public:
  Range_tracker(IRegion_manager * rm, std::string owner,std::string region_set_id) :
    _rm(rm), _owner(owner), _region_set_id(region_set_id)
  {
    assert(_rm);
    _rm->add_ref();   
  }
  virtual ~Range_tracker() { _rm->release_ref(); }

  addr_t reuse_or_allocate(std::string region_id, size_t size) {

    /* TODO: check it doesn't exist */
    size_t bs = _rm->block_size();
    uint64_t nblocks = size / bs;
    if(size % bs) nblocks++;

    bool reused;
    addr_t vaddr = 0;
    IBlock_device * bd = _rm->reuse_or_allocate_region(nblocks, _owner, region_id, vaddr, reused);
    if(!bd) throw General_exception("%s: region exhausted", __PRETTY_FUNCTION__);
    _table.push_back({vaddr,vaddr+(bs*nblocks)-1,bd});
    PLOG("Range-tracker: reuse-or-allocate result 0x%lx-0x%lx %p",vaddr,vaddr+(bs*nblocks)-1,bd);
    return vaddr;
  }

  IBlock_device * lookup(addr_t vaddr, addr_t& lba) {

    for(range_t& r: _table) {
      if(vaddr >= r.start && vaddr <= r.end) {
        lba = (vaddr - r.start) / PAGE_SIZE;
        return r.blkitf;
      }
    }
    throw Logic_exception("%s: lookup failed", __PRETTY_FUNCTION__);
    return nullptr;
  }

  void dump_info() {
    PINF("--range tracker--");
    for(range_t& r: _table) {
      PINF("0x%lx -> 0x%lx %p", r.start, r.end, r.blkitf);
    }
    PINF("-----------------");
  }

private:
  typedef struct {
    addr_t start;
    addr_t end;
    IBlock_device * blkitf;
  } range_t;

  IRegion_manager *    _rm;
  std::string          _owner;
  std::string          _region_set_id;
  std::vector<range_t> _table;
};

Pmem_fixed_component::
Pmem_fixed_component(std::string owner_id,
                     IBlock_device * block_device,
                     bool force_init) : _owner_id(owner_id),
                                                     _block_device(block_device)
{
  if(!block_device) throw API_exception("%s: IBlock_device interface param invalid", __PRETTY_FUNCTION__);
  block_device->add_ref();
  
  /* create region manager */
  IBase * comp = load_component("libcomanche-partregion.so",
                                Component::part_region_factory);
  assert(comp);
  IRegion_manager_factory* fact =
    (IRegion_manager_factory *) comp->query_interface(IRegion_manager_factory::iid());
      
  assert(fact);

  int flags = force_init ? IRegion_manager_factory::FLAGS_FORMAT : 0;
  _rm = fact->open(block_device, flags); 
  if(!_rm) throw General_exception("%s: unable to create region manager",
                                   __PRETTY_FUNCTION__);
  fact->release_ref();

  VOLUME_INFO vi;
  _block_device->get_volume_info(vi);
  _block_size = vi.block_size;
  
  /* open XMS module */
  _fd_xms = ::open("/dev/xms", O_RDWR, 0666);
  if(_fd_xms == -1) throw Constructor_exception("unable to open XMS module");
  PLOG("XMS module open OK.");
}

Pmem_fixed_component::
Pmem_fixed_component(std::string owner_id,
                     Component::IRegion_manager * rm,
                     bool force_init)
  : _owner_id(owner_id)
{
  if(!rm) throw API_exception("%s: IRegion_manager interface param invalid", __PRETTY_FUNCTION__);

  _rm = rm;
  _rm->add_ref();
  _block_device = _rm->get_underlying_block_device();
  _block_device->add_ref();
  assert(_block_device);
  _block_size = _rm->block_size();
  
  /* open XMS module */
  _fd_xms = ::open("/dev/xms", O_RDWR, 0666);
  if(_fd_xms == -1) throw Constructor_exception("unable to open XMS module");
  PLOG("XMS module open OK.");
}


Pmem_fixed_component::
~Pmem_fixed_component()
{
  if(_block_device)
    _block_device->release_ref();
  ::close(_fd_xms);
}


IPersistent_memory::pmem_t
Pmem_fixed_component::
open(std::string id, size_t size, int numa_node, bool& reused, void*& vptr)
{
  size = round_up_page(size);

  /* allocate memory */
  if(option_DEBUG) {
    PLOG("Pmem-fixed: allocating %ld MiB (numa node=%d)", REDUCE_MB(size), numa_node);
  }
  
  io_buffer_t iob = _block_device->allocate_io_buffer(size, PAGE_SIZE, numa_node);
  assert(iob);
  addr_t paddr = _block_device->phys_addr(iob);
  assert(paddr);

  memset(_block_device->virt_addr(iob),0xe,size); // TESTING ONLY
  
  addr_t vaddr = 0;
  assert(size % _block_size == 0);
  
  /* lookup persistent-virtual region */
  PLOG("Pmem-fixed: allocating %ld bytes (%ld blocks) from Region Manager identified by '%s'",
       size, size / _block_size, id.c_str());
  
  IBlock_device * bd = _rm->reuse_or_allocate_region(size / _block_size,
                                                     _owner_id,
                                                     id,
                                                     vaddr,
                                                     reused);
  assert(vaddr);

  PLOG("Pmem-fixed: region %s reused=%d", id.c_str(), reused);

  /* read in data */
  bd->read(iob, 0, 0 /* lba */ , size / _block_size /* lba count */);
  
  /* map region to physical memory */
  void  * ptr = ::mmap((void*) vaddr,
                       size,
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_FIXED,
                       _fd_xms,
                       (off_t) paddr);  // offset in the file is physical address

  if(ptr != ((void*)vaddr))
    throw General_exception("%s: mmap failed in allocate:%d addr=%p",
                            __PRETTY_FUNCTION__, errno, vaddr);

  PLOG("XMS: mmap returned %p", ptr);
  vptr = ptr;
  
  /* create handle */
  struct mem_handle * h = new mem_handle;
  h->ptr = ptr;
  h->size = size;
  h->iob = iob;
  h->bd = bd;

  std::lock_guard<std::mutex> g(_handle_list_lock);
  _handle_list.push_back(h);

  return h;
}

/** 
 * Free a previously allocated persistent memory region
 * 
 * @param handle Handle to persistent memory region
 */
void
Pmem_fixed_component::
close(IPersistent_memory::pmem_t handle, int flags)
{
  io_buffer_t iob;
  if(handle==nullptr)
    throw API_exception("%s: bad parameter", __PRETTY_FUNCTION__);

  struct mem_handle * h = static_cast<struct mem_handle*>(handle);

  if(!(flags & IPersistent_memory::FLAGS_NOFLUSH)) {
    this->persist(handle);
  }

  _block_device->free_io_buffer(h->iob);
  assert(h);
  assert(h->bd);
  h->bd->release_ref();

  
  std::lock_guard<std::mutex> g(_handle_list_lock);
  _handle_list.remove(h);
  delete h;
}

/** 
 * Explicitly erase region
 * 
 * @param handle 
 * 
 */
void
Pmem_fixed_component::
erase(IPersistent_memory::pmem_t handle)
{
  throw API_exception("not implemented");
}

/** 
 * Determine if a virtual address is in persistent memory
 * 
 * @param p Virtual address pointer
 * 
 * @return True if belonging to persistent memory
 */
bool
Pmem_fixed_component::
is_pmem(void * p)
{
  if(p==nullptr)
    throw API_exception("%s: bad parameter", __PRETTY_FUNCTION__);

  addr_t pa = reinterpret_cast<addr_t>(p);
  std::lock_guard<std::mutex> g(_handle_list_lock);
  for(auto& e: _handle_list) {
    if((pa >= (addr_t)e->ptr) &&
       (pa < (((addr_t)e->ptr)+e->size)))
      return true;    
  }
    
  return false;
}

/** 
 * Flush all volatile data to peristent memory in a non-transacation context.
 * 
 */
void
Pmem_fixed_component::
persist(pmem_t handle)
{
  struct mem_handle * h = static_cast<struct mem_handle*>(handle);
  if(h==nullptr ||
     h->bd == nullptr ||
     h->size == 0 ||
     h->iob == 0)
    throw API_exception("%s: bad handle", __PRETTY_FUNCTION__);

  /* synchronously write out whole block */
  size_t nblocks = h->size / _block_size;
  if(h->size % _block_size) nblocks++;

  PMAJOR("writing %ld blocks", nblocks);
  h->bd->write(h->iob, 0, 0, nblocks);
}

void
Pmem_fixed_component::
persist_scoped(pmem_t handle, void *ptr, size_t size)
{
  struct mem_handle * h = static_cast<struct mem_handle*>(handle);
  if(h==nullptr ||
     h->bd == nullptr ||
     h->ptr == nullptr ||
     h->size == 0 ||
     h->iob == 0)
    throw API_exception("%s: bad handle", __PRETTY_FUNCTION__);

  addr_t vaddr = round_down(reinterpret_cast<addr_t>(ptr), _block_size);
  assert(vaddr);

  if(!(((addr_t)ptr >= vaddr) && (((addr_t)ptr) <= (vaddr+size))))
    throw API_exception("%s: out of bounds", __PRETTY_FUNCTION__);
  
  addr_t iob_v_base = (addr_t) h->ptr;
  assert(iob_v_base);
  assert(vaddr >= iob_v_base);
  size_t offset = vaddr - iob_v_base;
  size_t lba_offset = offset / _block_size;
  if(offset % _block_size) lba_offset++;
  
  size_t nblocks = size / _block_size;
  if(size % _block_size) nblocks++;
  
  /* synchronously write out whole block */
  h->bd->write(h->iob, offset, lba_offset, nblocks); 
}

size_t
Pmem_fixed_component::
get_size(pmem_t handle)
{
  struct mem_handle * h = static_cast<struct mem_handle*>(handle);
  if(h==nullptr ||
     h->bd == nullptr ||
     h->ptr == nullptr ||
     h->size == 0 ||
     h->iob == 0)
    throw API_exception("%s: bad handle", __PRETTY_FUNCTION__);
  return h->size;
}

/** 
 * Start transaction
 * 
 */
void
Pmem_fixed_component::
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
Pmem_fixed_component::
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
  if(component_id == Pmem_fixed_component_factory::component_id()) {
    return static_cast<void*>(new Pmem_fixed_component_factory());
  }
  else
    return NULL;
}

