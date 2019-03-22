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


/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#include <assert.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/spinlocks.h>
#include <signal.h>
#include <boost/icl/split_interval_map.hpp>
#include <iostream>
#include <map>
#include <stdexcept>
#include <utility>

#include "tx_cache.h"

using namespace boost::icl;

struct alloc_info_t {
  size_t n_pages;
  size_t page_size;
};

typedef boost::icl::split_interval_set<addr_t> interval_set_t;
typedef interval_set_t::interval_type          ival_t;

static struct {
  Common::Ticket_lock            spin_lock;
  struct sigaction               default_sa;
  unsigned                       debug_level;
  interval_set_t                 intervals;
  std::map<void *, alloc_info_t> allocations;
  size_t                         mapped_page_count;
} tx_cache;

class Cache_state_guard {
 public:
  Cache_state_guard() { tx_cache.spin_lock.lock(); }
  ~Cache_state_guard() { tx_cache.spin_lock.unlock(); }
};

static void __segv_handler_trampoline(int sig, siginfo_t *si, void *context)
{
  addr_t fault_addr = reinterpret_cast<addr_t>(si->si_addr);

  if (sig != 11)
    throw Logic_exception("SIGSEGV handler received non-SEGV signal");

  {
    Cache_state_guard g;

    if (tx_cache.debug_level > 1)
      PMAJOR("[TX]: handler trampoline (fault=%lx, mapped_pages=%lu, errno=%d)",
             fault_addr, tx_cache.mapped_page_count, si->si_errno);

    if (tx_cache.debug_level > 2) {
      PLOG("[TX]: # intervals %lu", tx_cache.intervals.iterative_size());
      PLOG("[TX]: # allocations %lu", tx_cache.allocations.size());
      for (const ival_t &w : tx_cache.intervals) {
        PLOG("[TX]: interval:0x%lx-0x%lx", w.lower(), w.upper());
      }
    }

    auto i = tx_cache.intervals.find(fault_addr);
    if (i != tx_cache.intervals.end()) {
      /* map in a single page */
      void *range_start = (void *) i->lower();

      auto page_size = tx_cache.allocations[range_start].page_size;
      assert(page_size == KiB(4) || page_size == MiB(2));

      int   huge_flags = page_size == MB(2) ? MAP_HUGETLB | MAP_HUGE_2MB : 0;
      void *page_start = round_down(si->si_addr, page_size);

      void *np =
          mmap(page_start, page_size, PROT_READ | PROT_WRITE | PROT_EXEC,
               MAP_SHARED | MAP_FIXED | MAP_ANONYMOUS | huge_flags, 0, 0);
      if (np != page_start)
        throw Logic_exception("mapping of physical page failed (%s)",
                              strerror(errno));

      /* simulate memcpy */
      memset(np, 0, page_size);

      tx_cache.mapped_page_count++;
      if (tx_cache.debug_level > 2)
        PLOG("[TX]: mapped physical page (%lu) to address %p", page_size,
             page_start);
      return;
    }
    else {
      if (tx_cache.debug_level > 2) PLOG("calling default SIGSEGV handler");
    }
  }

  /* call default handler */
  tx_cache.default_sa.sa_sigaction(sig, si, context);
}

__attribute__((constructor)) static void tx_cache_ctor()
{
  tx_cache.debug_level = 0;

  /* attach SEGV handler */
  struct sigaction new_sa;
  new_sa.sa_flags = SA_SIGINFO;
  sigemptyset(&new_sa.sa_mask);
  new_sa.sa_sigaction = __segv_handler_trampoline;
  if (sigaction(SIGSEGV, &new_sa, &tx_cache.default_sa) == -1)
    throw std::logic_error("sigaction installing new handler failed");

  if (tx_cache.debug_level > 2) PINF("[TX]: installed new SIGSEGV handler.");

  tx_cache.mapped_page_count = 0;
}

namespace nupm
{
void *allocate_virtual_pages(size_t n_pages, size_t page_size, uint64_t hint)
{
  if (page_size != KB(4) && page_size != MB(2))
    throw std::invalid_argument("page size should be 4KiB or 2MiB");

  int huge_flags = 0;
  if (page_size == MB(2)) huge_flags = MAP_HUGETLB | MAP_HUGE_2MB;

  void *p = mmap(reinterpret_cast<void *>(hint), /* address hint */
                 page_size * n_pages, PROT_NONE,
                 MAP_SHARED | MAP_NORESERVE | MAP_ANONYMOUS | huge_flags,
                 0,  /* file */
                 0); /* offset */
  if (p == nullptr) throw Logic_exception("mmap failed unexpectedly");

  addr_t paddr = (addr_t) p;
  {
    Cache_state_guard g;
    const auto ival = ival_t::closed(paddr, paddr + (n_pages * page_size));
    PLOG("TX: add interval %lx-%lx", paddr, paddr + (n_pages * page_size));
    tx_cache.intervals.add(ival);
    assert(page_size > 0);
    tx_cache.allocations[p] = {n_pages, page_size};
  }

  return p;
}

int free_virtual_pages(void *p)
{
  PLOG("[TX]: free_virtual_pages(%p)", p);
  Cache_state_guard g;

  auto i = tx_cache.allocations.find(p);
  if (i == tx_cache.allocations.end())
    throw std::invalid_argument("bad allocation pointer");

  auto n_pages   = i->second.n_pages;
  auto page_size = i->second.page_size;
  assert(page_size > 0);
  assert(n_pages > 0);

  addr_t paddr = (addr_t) p;
  auto   ival  = ival_t::closed(paddr, paddr + (n_pages * page_size));
  tx_cache.intervals.erase(ival);

  int rc = munmap(p, n_pages * page_size);
  tx_cache.allocations.erase(i);
  tx_cache.mapped_page_count -= n_pages;

  return rc;
}
}  // namespace nupm
