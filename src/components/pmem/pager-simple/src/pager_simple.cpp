#include <sys/mman.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <iostream>
#include <cstring>

#include <common/cpu.h>
#include <api/region_itf.h>
#include "pager_simple.h"

//#define DISABLE_IO /* TESTING ONLY */

using namespace Component;


/** 
 * Map vaddr to a block device
 * 
 * @param rm 
 * @param owner 
 * @param heap_set_id 
 */
class Range_tracker
{
public:
  Range_tracker(IRegion_manager * rm, std::string owner,std::string heap_set_id) :
    _rm(rm), _owner(owner), _heap_set_id(heap_set_id)
  {
    assert(_rm);
    _rm->add_ref();   
  }
  virtual ~Range_tracker() { _rm->release_ref(); }

  addr_t reuse_or_allocate(std::string region_id, size_t size, bool& reused) {

    /* TODO: check it doesn't exist */
    size_t bs = _rm->block_size();
    uint64_t nblocks = size / bs;
    if(size % bs) nblocks++;

    addr_t vaddr;
    IBlock_device * bd = _rm->reuse_or_allocate_region(nblocks,
                                                       _owner,
                                                       region_id,
                                                       vaddr,
                                                       reused);
    if(!bd) throw General_exception("%s: region exhausted", __PRETTY_FUNCTION__);
    _table.push_back({vaddr,vaddr+(bs*nblocks)-1,bd});
    PLOG("Range-tracker: reuse-or-allocate result 0x%lx-0x%lx %p",vaddr,vaddr+(bs*nblocks)-1,bd);
    return vaddr;
  }

  IBlock_device * lookup(addr_t vaddr, addr_t& lba) {

    if(!vaddr)
      throw API_exception("%s: bad param", __PRETTY_FUNCTION__);
    
    for(range_t& r: _table) {
      if(vaddr >= r.start && vaddr <= r.end) {
        lba = (vaddr - r.start) / PAGE_SIZE;
        return r.blkitf;
      }
    }
    throw Logic_exception("%s: lookup failed (vaddr=%lx)", __PRETTY_FUNCTION__, vaddr);
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

  IRegion_manager * _rm;
  std::string _owner;
  std::string _heap_set_id;
  std::vector<range_t> _table;
};

Simple_pager_component::
Simple_pager_component(size_t nr_pages,
                       std::string heap_set_id,
                       Component::IBlock_device * block_device,
                       bool force_init)
  : _block_dev(block_device), _rm(nullptr), _heap_set_id(heap_set_id)
{
  if(!_block_dev) throw API_exception("%s: invalid param", __PRETTY_FUNCTION__);

  /* create region manager */
  IBase * comp = load_component("libcomanche-partregion.so",
                                Component::part_region_factory);
  assert(comp);
  IRegion_manager_factory* fact =
    (IRegion_manager_factory *) comp->query_interface(IRegion_manager_factory::iid());
      
  assert(fact);

  if(force_init)
    _rm = fact->open(_block_dev, IRegion_manager_factory::FLAGS_FORMAT);
  else
    _rm = fact->open(_block_dev, 0);
  
  if(!_rm) throw General_exception("%s: unable to create region manager",
                                   __PRETTY_FUNCTION__);
  fact->release_ref();

  /* initialize memory */
  init_memory(nr_pages);

  /* initialize range tracker */
  _tracker = new Range_tracker(_rm, "myTiger", heap_set_id);

  PINF("Part-region component loaded OK.");
}

Simple_pager_component::~Simple_pager_component()
{
  /* flush active pages */  
  for(unsigned slot=0;slot<_nr_pages;slot++) {
    if(_pages[slot].vaddr) {
      uint64_t buffer_offset = PAGE_SIZE * slot;
      addr_t lba;
      IBlock_device * bd = _tracker->lookup(_pages[slot].vaddr, lba);
      assert(bd);
      bd->write(_iob, buffer_offset, lba, 1);
    }
  }
  
  delete _tracker;

  _rm->release_ref();
  _block_dev->release_ref();
  //  delete _pages;
}

void
Simple_pager_component::
init_memory(size_t nr_pages)
{
  PLOG("Initializing simple-pager ..");
  
  if(nr_pages == 0)
    throw API_exception("invalid param:%s",__FUNCTION__);

  _nr_pages = nr_pages;

  /* set up IO memory */
  _iob = _block_dev->allocate_io_buffer(PAGE_SIZE * nr_pages,
                                        PAGE_SIZE,
                                        NUMA_NODE_ANY);
  _phys_base = _block_dev->phys_addr(_iob);
  assert(_phys_base > 0);

  addr_t phys = _phys_base;
  _pages = new struct page_descriptor [_nr_pages];
  
  for(unsigned i=0;i<nr_pages;i++) {
    _pages[i].gwid  = 0;
    _pages[i].paddr  = phys;
    _pages[i].vaddr  = 0;
    //    _pages.push_back(phys);
    PLOG("[simple-pager]: added phys page %lx", phys);
    phys+= PAGE_SIZE;
  }

  if(option_DEBUG) {
    PINF("[simple-pager]: allocated %ld pages starting at %lx",
         _nr_pages,_phys_base);
  }
}

void *
Simple_pager_component::
get_region(std::string id, size_t size, bool& reused)
{
  return reinterpret_cast<void*>(_tracker->reuse_or_allocate(id, size, reused));
}

void
Simple_pager_component::
request_page(addr_t virt_addr_faulted,
             addr_t *out_phys_addr_map,
             addr_t *out_virt_addr_evict)
{
  PINF("PF#: virt=%lx", virt_addr_faulted);
  
  if(virt_addr_faulted == 0)
    throw General_exception("SIGEV on NULL pointer");
  
  assert(virt_addr_faulted);
  assert(out_phys_addr_map);
  assert(out_virt_addr_evict);

  if(option_DEBUG) {
    PLOG("request page: fault=%lx", virt_addr_faulted);
  }
  
  /* select page to evict */
  unsigned slot = _request_num % _nr_pages;
  _request_num++;

  //#define FORCE_SYNC // TESTING ONLY
#ifndef FORCE_SYNC // TESTING ONLY
  if(_pages[slot].gwid) {
#ifndef DISABLE_IO
    //    PLOG("waiting for completion gwid=%ld",_pages[slot].gwid);
    while(!_block_dev->check_completion(_pages[slot].gwid)) {
      cpu_relax();
    }
    _pages[slot].gwid = 0;
#endif
    if(option_DEBUG)
      PLOG("swapping out: vaddr evict=%p", (void*) *out_virt_addr_evict);
  }
#endif
  *out_virt_addr_evict = _pages[slot].vaddr;

  
#ifndef DISABLE_IO
  addr_t lba;
  IBlock_device * bd = nullptr;
  uint64_t buffer_offset = PAGE_SIZE * slot;
  /* swap out */
  {
    if(_pages[slot].vaddr > 0) {
      bd = _tracker->lookup(_pages[slot].vaddr, lba);
      
      /* issue async write-back */
#ifndef FORCE_SYNC // TESTING only.
      _pages[slot].gwid = bd->async_write(_iob, buffer_offset, lba,1);
#else
      bd->write(_iob, buffer_offset, lba, 1);
#endif
    }
  }
  /* swap in */
  {
    //    _tracker->dump_info();
    /* issue synchronous read */
    bd = _tracker->lookup(virt_addr_faulted, lba);
    if(option_DEBUG)
      PLOG("swapping in: vaddr=0x%lx lba=0x%lx", virt_addr_faulted, lba);

    bd->read(_iob, buffer_offset, lba, 1);
  }
#endif
  
  _pages[slot].vaddr = virt_addr_faulted;
  *out_phys_addr_map = _pages[slot].paddr;
  assert(*out_phys_addr_map > 0);
  
  if(option_DEBUG) {
     PLOG("resolve pf response: phys=%lx evict=%lx", *out_phys_addr_map, *out_virt_addr_evict);
  }
}


void
Simple_pager_component::
clear_mappings(addr_t vaddr, size_t size)
{
  addr_t vaddr_end = vaddr + size;
  for(unsigned i=0;i<_nr_pages;i++) {
    if(_pages[i].vaddr >= vaddr && _pages[i].vaddr < vaddr_end) {

      if(_pages[i].vaddr && _pages[i].gwid) {
#ifndef DISABLE_IO
        while(!_block_dev->check_completion(_pages[i].gwid)) {
          //          PWRN("waiting on completions: queue too short?");
          cpu_relax();
        }
#endif
      }

      _pages[i].gwid  = 0;
      _pages[i].vaddr  = 0;
    }
  }
}

void
Simple_pager_component::
flush(addr_t vaddr, size_t size)
{
  assert(vaddr);
  assert(size);
  
  for(unsigned slot=0;slot<_nr_pages;slot++) {
    if(_pages[slot].vaddr &&
       (vaddr >= _pages[slot].vaddr) &&
       (vaddr < (_pages[slot].vaddr + PAGE_SIZE)))
    {
      uint64_t buffer_offset = PAGE_SIZE * slot;
      addr_t lba;
      IBlock_device * bd = _tracker->lookup(_pages[slot].vaddr, lba);
      assert(bd);
      bd->write(_iob, buffer_offset, lba, 1);
    }
  }    
}

#undef DISABLE_IO

/** 
 * Factory entry point
 * 
 * load component will create a instance of current component
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{

  if(component_id == Simple_pager_component_factory::component_id()) {
    return static_cast<void*>(new Simple_pager_component_factory());
  }
  else{
    return NULL;
  }
}

