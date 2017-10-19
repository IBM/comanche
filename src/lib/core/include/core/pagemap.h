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

#ifndef __CORE_PAGEMAP_H__
#define __CORE_PAGEMAP_H__

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <fstream>
#include <sstream>
#include <vector>
#include <assert.h>
#include <unistd.h>
#include <iostream>

#include <common/types.h>
#include <common/logging.h>
#include <common/utils.h>

namespace Core
{
class Pagemap
{
 private:
  static constexpr bool option_DEBUG = true;

  int    _fd_pagemap;
  int    _fd_kpageflags;
  int    _page_size;
  size_t _page_shift;

 public:
  enum page_flag_t {
    PAGE_FLAG_LOCKED     = (1ULL << 0),
    PAGE_FLAG_ERROR      = (1ULL << 1),
    PAGE_FLAG_REFERENCED = (1ULL << 2),
    PAGE_FLAG_UPTODATE   = (1ULL << 3),
    PAGE_FLAG_DIRTY      = (1ULL << 4),
    PAGE_FLAG_LRU        = (1ULL << 5),
    PAGE_FLAG_ACTIVE     = (1ULL << 6),
    PAGE_FLAG_SLAB       = (1ULL << 7),
    PAGE_FLAG_WRITEBACK  = (1ULL << 8),
    PAGE_FLAG_RECLAIM    = (1ULL << 9),
    PAGE_FLAG_BUDDY      = (1ULL << 10),
    /* kernel > 2.6.31 */
    PAGE_FLAG_MMAP          = (1ULL << 11),
    PAGE_FLAG_ANON          = (1ULL << 12),
    PAGE_FLAG_SWAPCACHE     = (1ULL << 13),
    PAGE_FLAG_SWAPBACKED    = (1ULL << 14),
    PAGE_FLAG_COMPOUND_HEAD = (1ULL << 15),
    PAGE_FLAG_COMPOUND_TAIL = (1ULL << 16),
    PAGE_FLAG_HUGE          = (1ULL << 17),
    PAGE_FLAG_UNEVICTABLE   = (1ULL << 18),
    PAGE_FLAG_HWPOISON      = (1ULL << 19),
    PAGE_FLAG_NOPAGE        = (1ULL << 20),
    PAGE_FLAG_KSM           = (1ULL << 21),
    PAGE_FLAG_THP           = (1ULL << 22),
  };

  struct range_t {
    addr_t start;
    addr_t end;

    range_t(addr_t s, addr_t e) : start(s), end(e)
    {
    }
  };

  /** 
     * Read in the current memory regions from /proc/self/maps.
     * 
     * @param range_list [out] vector of range_t elements
     */
  static void read_regions(std::vector<range_t>& range_list);

 public:
  typedef uint64_t pfn_t;
  typedef uint64_t pmentry_t;



 public:
  /** 
     * Constructor; use calling process id
     * 
     */
  Pagemap()
  {
    _fd_pagemap = open("/proc/self/pagemap", O_RDONLY);
    assert(_fd_pagemap != -1);

    _fd_kpageflags = open("/proc/kpageflags", O_RDONLY);
    assert(_fd_kpageflags);

    _page_shift = PAGE_SHIFT;
    _page_size  = 1 << PAGE_SHIFT;
  }

  ~Pagemap()
  {
    close(_fd_pagemap);
    close(_fd_kpageflags);
  }


  /** 
     * Translate virtual to physical address for the current process.
     * 
     * 
     * @return 
     */
  addr_t virt_to_phys(void* vaddr)
  {
    pfn_t pfn = get_page_frame_number(vaddr);
    return (pfn << _page_shift) + (((addr_t) vaddr) % _page_size);
  }

  /** 
     * Look up the physical Page Frame Number from the virtual address
     * 
     * @param vaddr Virtual address 
     * 
     * @return Page Frame Number if present
     */
  pfn_t get_page_frame_number(void* vaddr);


  /** 
     * Get pagemap entry for given address
     * 
     * @param vaddr Address (not range checked)
     * 
     * @return 
     */
  pmentry_t get_page_entry(void* vaddr);


  /** 
     * Get a full set of entries for a given region.
     * 
     * @param region Region address
     * @param region_size Size of region in bytes
     * 
     * @return Pointer to ::malloc'ed memory which client must free.
     */
  pmentry_t* get_page_entries(addr_t start_page, addr_t end_page, size_t* out_num_entries);

  /** 
     * Return soft-dirty state: see https://www.kernel.org/doc/Documentation/vm/pagemap.txt
     * 
     * @param pme Pagemap entry
     * 
     * @return True if set
     */
  bool is_soft_dirty(pmentry_t pme)
  {
    return pme & (1UL << 55);
  }

  /** 
     * Return the page flags for a given physical page
     * 
     * @param vaddr Virtual address mapped to desired physical page
     * 
     * @return 64-bit page flags
     */
  page_flag_t page_flags(void* vaddr)
  {
    pfn_t    pfn   = get_page_frame_number(vaddr);
    uint64_t entry = 0;

    int rc = pread(_fd_kpageflags, (void*) &entry, 8, pfn * 8);
    if (rc < 0) {
      perror("page_flags pread failed: did you run as root?");
      return (page_flag_t) 0;
    }

    return (page_flag_t) entry;
  }

  /** 
     * Return the page flags for a given physical page
     * 
     * @param vaddr Virtual address mapped to desired physical page
     * 
     * @return 64-bit page flags
     */
  page_flag_t page_flags(uint64_t pfn)
  {
    uint64_t entry = 0;

    int rc = pread(_fd_kpageflags, (void*) &entry, 8, pfn * 8);
    if (rc < 0) {
      perror("page_flags pread failed: did you run as root?");
      return (page_flag_t) 0;
    }

    return (page_flag_t) entry;
  }


  /** 
     * Return the size of the file
     * 
     * 
     * @return Size of the file in bytes
     */
  size_t file_size()
  {
    struct stat buf;
    fstat(_fd_pagemap, &buf);
    return buf.st_size;
  }

  /** 
     * Dump page map information for the calling process
     * 
     */
  void dump_self_region_info();

  /** 
     * Dump some debugging information from /proc/self/maps
     * 
     */
  void dump()
  {
    using namespace std;
    size_t s = file_size();
#if (__SIZEOF_SIZE_T__ == 4)
    printf("Pagemap size:%d bytes\n", s);
#else
    printf("Pagemap size:%ld bytes\n", s);
#endif

    /* retrieve the list of ranges from /proc/self/maps */
    vector<range_t> range_list;
    read_regions(range_list);

    for (vector<range_t>::iterator i = range_list.begin(); i != range_list.end(); i++) {
      struct range_t r = *i;
      while (r.start != r.end) {
        printf("PAGE: 0x%lx\n", r.start);
        r.start += 0x1000;  // 4K
      }
    }
  }


  void dump(addr_t region_start, size_t region_size)
  {
    using namespace std;
    size_t s = file_size();
#if (__SIZEOF_SIZE_T__ == 4)
    printf("Pagemap size:%d bytes\n", s);
#else
    printf("Pagemap size:%ld bytes\n", s);
#endif

    /* retrieve the list of ranges from /proc/self/maps */
    vector<range_t> range_list;
    read_regions(range_list);

    for (vector<range_t>::iterator i = range_list.begin(); i != range_list.end(); i++) {
      struct range_t r = *i;
      while (r.start != r.end) {
        if (r.start >= region_start && r.start < (region_start + region_size)) {
          printf("PAGE: 0x%lx\n", r.start);
        }
        r.start += 0x1000;  // 4K
      }
    }
  }

  /** 
     * Dump out page flag debugging info
     * 
     * @param f 64-bit flag
     */
  void dump_page_flags(page_flag_t f)
  {
    if (f & PAGE_FLAG_LOCKED) PLOG("pageflag:locked");
    if (f & PAGE_FLAG_ERROR) PLOG("pageflag:error");
    if (f & PAGE_FLAG_REFERENCED) PLOG("pageflag:referenced");
    if (f & PAGE_FLAG_UPTODATE) PLOG("pageflag:uptodate");
    if (f & PAGE_FLAG_DIRTY) PLOG("pageflag:dirty");
    if (f & PAGE_FLAG_LRU) PLOG("pageflag:lru");
    if (f & PAGE_FLAG_ACTIVE) PLOG("pageflag:active");
    if (f & PAGE_FLAG_SLAB) PLOG("pageflag:slab");
    if (f & PAGE_FLAG_WRITEBACK) PLOG("pageflag:writeback");
    if (f & PAGE_FLAG_RECLAIM) PLOG("pageflag:reclaim");
    if (f & PAGE_FLAG_BUDDY) PLOG("pageflag:buddy");
    if (f & PAGE_FLAG_MMAP) PLOG("pageflag:mmap");
    if (f & PAGE_FLAG_ANON) PLOG("pageflag:anon");
    if (f & PAGE_FLAG_SWAPCACHE) PLOG("pageflag:swapcache");
    if (f & PAGE_FLAG_SWAPBACKED) PLOG("pageflag:swapbacked");
    if (f & PAGE_FLAG_COMPOUND_HEAD) PLOG("pageflag:compound head");
    if (f & PAGE_FLAG_COMPOUND_TAIL) PLOG("pageflag:compound tail");
    if (f & PAGE_FLAG_HUGE) PLOG("pageflag:huge");
    if (f & PAGE_FLAG_UNEVICTABLE) PLOG("pageflag:unevictable");
    if (f & PAGE_FLAG_HWPOISON) PLOG("pageflag:hwpoison");
    if (f & PAGE_FLAG_NOPAGE) PLOG("pageflag:nopage");
    if (f & PAGE_FLAG_KSM) PLOG("pageflag:ksm");
    if (f & PAGE_FLAG_THP) PLOG("pageflag:thp");
  }
};
}

#endif  // __CORE_PAGEMAP_H__
