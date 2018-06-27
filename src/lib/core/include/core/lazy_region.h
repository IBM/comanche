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

#ifndef __COMMON_LAZY_REGION_H__
#define __COMMON_LAZY_REGION_H__

#include <list>
#include <mutex>
#include <signal.h>
#include <sys/mman.h>

#include <common/utils.h>
#include <common/assert.h>
#include <common/logging.h>
#include <common/exceptions.h>
#include <common/types.h>

namespace Core
{
namespace Slab
{
// forward decls
//
class Lazily_extending_region;

/** 
     * Interval map - used to map addr to instance of Lazily_extending_region class
     * 
     */
struct __map_entry {
  addr_t                   start;
  addr_t                   end;
  Lazily_extending_region *inst;
};

static std::list<__map_entry> _interval_map;  // we should use a tree
static std::mutex             _interval_map_lock;

/** 
     * Add interval to map; used to look up class instance from SIGSEGV handler
     * 
     * @param start Start address
     * @param end End address
     * @param inst Instance pointer
     */
static void __add_interval(addr_t start, addr_t end, Lazily_extending_region *inst)
{
  std::lock_guard<std::mutex> lock(_interval_map_lock);
  for (auto i = _interval_map.begin(); i != _interval_map.end(); i++) {
    if (start < i->start) {
      _interval_map.insert(i, {start, end, inst});
      break;
    }
  }
  _interval_map.push_back({start, end, inst});
  //PDBG("added interval (%lx-%lx)", start, end);
}

/** 
     * Lookup instance for a given address
     * 
     * @param addr 
     * 
     * @return 
     */
static Lazily_extending_region *__lookup_inst(addr_t addr)
{
  assert(addr > 0);
  std::lock_guard<std::mutex> lock(_interval_map_lock);
  for (auto i : _interval_map) {
    if (addr >= i.start && addr <= i.end) return i.inst;
  }
  PERR("address (%p) not found in interval map", (void *) addr);
  assert(0);
  return NULL;  // not found
}


/** 
     * Class to manage a lazily extending region of memory
     * 
     */
class Lazily_extending_region
{
 private:
  static constexpr bool option_DEBUG = false;

 public:
  /** 
       * Constructor
       * 
       * @param size Maximum memory allocation 
       */
  Lazily_extending_region(size_t size) : _max_size(size), _mapped_size(0)
  {
    assert(size % PAGE_SIZE == 0);

    // set up SIGSEGV handler
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = SIGSEGV_handler;

    if (sigaction(SIGSEGV, &sa, NULL) == -1) throw Constructor_exception("unable to set SIGSEGV handler");

    static addr_t addr_hint = PREFERRED_VADDR;

    _ptr = mmap((void *) (addr_hint), _max_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, 0, 0);
    addr_hint += 0x10000000ULL;

    if (option_DEBUG) {
      PLOG("mmap allocated region (%p)", _ptr);
    }

    __add_interval((addr_t) _ptr, ((addr_t) _ptr) + _max_size, this);
  }

  /** 
       * Destructor
       * 
       */
  ~Lazily_extending_region() noexcept(false)
  {
    assert(_ptr);
    int rc = ::munmap(_ptr, _max_size);
    if (rc) throw General_exception("::munmap failed in Lazilty_extending_region dtor");
    assert(rc == 0);
  }

  /** 
       * Get base address of allocated region
       * 
       * 
       * @return Pointer to allocated region
       */
  addr_t base()
  {
    return (addr_t) _ptr;
  }

  /** 
       * Get pointer to allocated region
       * 
       * 
       * @return 
       */
  void *ptr() const
  {
    return _ptr;
  }

  /** 
       * Return amount of memory mapped
       * 
       * 
       * @return Size of used/memory in bytes
       */
  size_t mapped_size()
  {
    __sync_synchronize();
    return _mapped_size;
  }

  /** 
       * Return number of mapped pages
       * 
       * 
       * @return Number of mapped pages
       */
  size_t mapped_pages()
  {
    return mapped_size() / PAGE_SIZE;
  }

  /** 
       * Maximum size of the expanding region
       * 
       * 
       * @return 
       */
  size_t max_size()
  {
    __sync_synchronize();
    return _max_size;
  }

 private:
#if (__SIZEOF_POINTER__ == 8)
  static constexpr auto PREFERRED_VADDR = 0xBB00000000ULL;
#else
  static constexpr auto PREFERRED_VADDR = 0xBB000000UL;
#endif

  void * _ptr;
  size_t _max_size; /**< maximum size of the expanding slab */
  size_t _mapped_size; /**< size mapped to physical memory */

 private:
  /** 
       * Increment # of map pages
       * 
       */
  void increment_mapping()
  {
    _mapped_size += PAGE_SIZE;
    assert(_mapped_size <= _max_size);
    if (option_DEBUG) {
      PDBG("#pages mapped:%lu", _mapped_size / PAGE_SIZE);
    }
  }

  static addr_t round_down_page(addr_t a)
  {
    /* round up to 4K page */
    if ((a & ((addr_t) 0xfff)) == 0)
      return a;
    else
      return (a & (~((addr_t) 0xfff)));
  }

  /** 
       * SEGV signal handler
       * 
       * @param sig 
       * @param si 
       * @param context 
       */
  static void SIGSEGV_handler(int /*sig*/, siginfo_t *si, void * /*context*/)
  {
    // TODO throw the stock handler on SEGV outside of
    // our memory

    //        static addr_t _last_fault_addr = 0;

    void *faulting_page = (void *) round_down_page((addr_t) si->si_addr);
    if (option_DEBUG) {
      PDBG("fault addr:%p", faulting_page);
    }
    assert(faulting_page);

    // if(_last_fault_addr) // check for sequential faults
    //   assert((_last_fault_addr + PAGE_SIZE) == ((addr_t) faulting_page));

    // _last_fault_addr = ((addr_t) faulting_page);

    Lazily_extending_region *inst = __lookup_inst((addr_t) faulting_page);
    assert(inst);
    inst->increment_mapping();

    ::mprotect(faulting_page, PAGE_SIZE, PROT_READ | PROT_WRITE);
  }
};

}  // namespace Slab
}  // namespace Core

#endif
