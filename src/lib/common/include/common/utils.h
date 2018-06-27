/*
   eXokernel Development Kit (XDK)

   Samsung Research America Copyright (C) 2013

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.
*/


/*
  Authors:
  Copyright (C) 2016, Daniel G. Waddington <daniel.waddington@ibm.com>
  Copyright (C) 2013, Daniel G. Waddington <d.waddington@samsung.com>
*/

#ifndef __COMMON_UTILS_H__
#define __COMMON_UTILS_H__

#include <sys/stat.h>
#include <fcntl.h>
#include <stdarg.h>
#include <sys/time.h>
#include <unistd.h>
#include <zlib.h>

#include "cpu.h"
#include "cpu_bitset.h"
#include "types.h"

#if defined(__unix__)
#include <numa.h>
#endif

#define INLINE inline __attribute__((always_inline))

#ifndef SUPPRESS_NOT_USED_WARN
#define SUPPRESS_NOT_USED_WARN __attribute__((unused))
#endif

#ifndef PAGE_SIZE
#define PAGE_SIZE (4096UL)  // sysconf(_SC_PAGE_SIZE)
#endif

#ifndef PAGE_MASK
#define PAGE_MASK   (~(PAGE_SIZE - 1))
#endif

#ifndef PAGE_SHIFT
#define PAGE_SHIFT 12
#endif

#define HUGE_PAGE_SIZE (2 * 1024 * 1024UL)
#define HUGE_MAGIC 0x0fabf00dUL

#if defined(__cplusplus)
extern "C"
#endif
  void
  panic(const char* format, ...) __attribute__((format(printf, 1, 2)));

#ifndef assert
#ifdef CONFIG_DEBUG
#undef assert
#define assert(X) \
  if (!(X)) panic("assertion failed at %s:%d", __FILE__, __LINE__);
#else
#define assert(X)
#endif
#endif

#if defined(__x86_64__)
#define mb() asm volatile("mfence" ::: "memory")
#define rmb() asm volatile("lfence" ::: "memory")
#define wmb() asm volatile("sfence" ::: "memory")
#elif defined(__x86_32__)
#define mb() asm volatile("lock; addl $0,0(%%esp)", "mfence", (0 * 32 + 26))
#define rmb() asm volatile("lock; addl $0,0(%%esp)", "lfence", (0 * 32 + 26))
#define wmb() asm volatile("lock; addl $0,0(%%esp)", "sfence", (0 * 32 + 25))
#elif defined(__arm__)
#define mb() asm volatile("" ::: "memory")
#define rmb() asm volatile("" ::: "memory")
#define wmb() asm volatile("" ::: "memory")
#else
#error Memory barriers not implemented
#endif

#define REDUCE_KB(X) (X >> 10)
#define REDUCE_MB(X) (X >> 20)
#define REDUCE_GB(X) (X >> 30)
#define REDUCE_TB(X) (X >> 40)

#define KB(X) (X << 10)
#define MB(X) (X << 20)
#define GB(X) (((unsigned long)X) << 30)
#define TB(X) (((unsigned long)X) << 40)

#define REDUCE_KiB(X) (X >> 10)
#define REDUCE_MiB(X) (X >> 20)
#define REDUCE_GiB(X) (X >> 30)
#define REDUCE_TiB(X) (X >> 40)

#define KiB(X) (X << 10)
#define MiB(X) (X << 20)
#define GiB(X) (((unsigned long)X) << 30)
#define TiB(X) (((unsigned long)X) << 40)

#define PAGE_SHIFT_512 9
#define PAGE_SHIFT_4K 12
#define PAGE_SHIFT_2MB 21

/**
 * Forward align a pointer
 *
 * @param p Pointer to check
 * @param alignment Alignment in bytes
 *
 * @return Aligned pointer
 */
INLINE static void* align_pointer(void* p, unsigned long alignment)
{
  if (addr_t(p) % alignment == 0) return p;
  return reinterpret_cast<void*>((mword_t(p) & ~(mword_t(alignment) - 1UL)) + alignment);
}

/** 
 * Align (upward) address
 * 
 * @param p Address to align
 * @param alignment Alignment in bytes
 * 
 * @return Aligned address
 */
INLINE static addr_t round_up(addr_t p, unsigned long alignment)
{
  if (p % alignment == 0) return p;
  return addr_t((mword_t(p) & ~(mword_t(alignment) - 1UL)) + alignment);
}

/** 
 * Align (down) address
 * 
 * @param p Address to align
 * @param alignment Alignment in bytes
 * 
 * @return Aligned address
 */
INLINE static addr_t round_down(addr_t p, unsigned long alignment)
{
  if (p % alignment == 0) return p;
  return addr_t((mword_t(p) & ~(mword_t(alignment) - 1UL)));
}

/** 
 * Align (down) pointer
 * 
 * @param p Pointer to align
 * @param alignment Alignment in bytes
 * 
 * @return Aligned address
 */
INLINE static void* round_down(void* p, unsigned long alignment)
{
  if (mword_t(p) % alignment == 0) return p;
  return reinterpret_cast<void*>(mword_t(p) & ~(mword_t(alignment) - 1UL));
}


/**
 * Check pointer alignment
 *
 * @param p Pointer to check
 * @param alignment Alignment in bytes
 *
 * @return
 */
INLINE static bool check_aligned(void* p, unsigned alignment)
{
  return (!(reinterpret_cast<unsigned long>(p) & (alignment - 1UL)));
}

INLINE static bool check_aligned(unsigned long p, unsigned alignment)
{
  return (!(static_cast<unsigned long>(p) & (alignment - 1UL)));
}

/**
 * Checks whether or not the number is power of 2.
 */
INLINE static bool is_power_of_two(unsigned long x)
{
  return (x != 0) && ((x & (x - 1UL)) == 0);
}

/** 
 * Find N where 2^N = val
 * 
 * @param val 
 * 
 * @return 
 */
INLINE static unsigned log_base2(uint64_t val)
{
  return __builtin_ctzl(val);
}

/**
 * Round up to nearest 4MB aligned
 *
 */
INLINE static addr_t round_up_superpage(addr_t a)
{
  /* round up to 4MB super page */
  if ((a & addr_t(0x3fffff)) == 0)
    return a;
  else
    return (a & (~addr_t(0x3fffff))) + MB(4);
}

/**
 * Round up to nearest 4K aligned
 *
 * @param a
 *
 * @return
 */
INLINE addr_t round_up_page(addr_t a)
{
  /* round up to 4K page */
  if ((a & addr_t(0xfff)) == 0)
    return a;
  else
    return (a & (~addr_t(0xfff))) + KB(4);
}

/**
 * Round down to 4K page
 *
 * @param a
 *
 * @return Page address
 */
INLINE addr_t round_down_page(addr_t a)
{
  /* round down to 4K page */
  if ((a & addr_t(0xfff)) == 0)
    return a;
  else
    return (a & (~addr_t(0xfff)));
}

INLINE addr_t round_down_by_shift(addr_t a, unsigned shift)
{
  return ((a >> shift) << shift);
}

/**
 * Round down to cache line
 *
 * @param a
 *
 * @return Aligned address
 */
INLINE void* round_down_cacheline(void* a)
{
  return reinterpret_cast<void*>(addr_t(a) & ~(CACHE_LINE_SHIFT - 1));
}

/**
 * Round up to 2^N bytes
 *
 */
INLINE addr_t round_up_log2(addr_t a)
{
  int clzl  = __builtin_clzl(a);
  int fsmsb = int(((sizeof(addr_t) * 8) - clzl));
  if ((addr_t(1) << (fsmsb - 1)) == a) fsmsb--;

  return (addr_t(1) << fsmsb);
}

/** Returns current system time. */
INLINE static struct timeval now()
{
  struct timeval t;

  /**
   * Beware this is doing a system call.
   *
   */
  gettimeofday(&t, 0);
  return t;
}

/**
 * Returns the difference between two timestamps in seconds.
 * @param t1 one timestamp.
 * @param t2 the other timestamp.
 * @return the difference the two timestamps.
 */
INLINE static double operator-(const struct timeval& t1, const struct timeval& t2)
{
  return double(t1.tv_sec - t2.tv_sec) + 1.0e-6 * double(t1.tv_usec - t2.tv_usec);
}

#ifndef __CUDACC__
INLINE unsigned min(unsigned x, unsigned y)
{
  return x < y ? x : y;
}
#endif

#ifndef __CUDACC__
INLINE unsigned max(unsigned x, unsigned y)
{
  return x > y ? x : y;
}
#endif

/**
 * Touch memory at huge (2MB) page strides
 *
 * @param addr Pointer to memory to touch
 * @param size Size of memory in bytes
 */
void touch_huge_pages(void* addr, size_t size);

/**
 * Touch memory at 4K page strides
 *
 * @param addr Pointer to memory to touch
 * @param size Size of memory in bytes
 */
void touch_pages(void* addr, size_t size);

#if defined(__linux__)
/**
 * Get the cpu is from numa cpu mask
 *
 * @param mask Cpu bitmask from numa node
 * @param n The nth cpu in the numa node
 * @return The global cpu id
 */
INLINE unsigned get_cpu_id(struct bitmask* mask, unsigned n)
{
  uint64_t cpumask   = *(mask->maskp);
  unsigned curr_pos  = 0;
  unsigned remaining = n;

  do {
    if (cpumask & 1) {
      remaining--;
      if (remaining == 0) break;
    }
    cpumask = cpumask >> 1;
    curr_pos++;
  } while (remaining != 0);

  return curr_pos;
}
#endif

/**
 * Determines the actual system thread affinities from logical (consecutive)
 * affinities for a given NUMA node.
 * @param logical_affinities bitset with the logical thread affinities for the
 *                           given NUMA node.
 * @param numa_node ID of the NUMA node.
 * @return bitset with actual system affinities.
 */
Cpu_bitset get_actual_affinities(const Cpu_bitset& logical_affinities, const int numa_node);

#if defined(__i386__) || defined(__x86_64__)
#define cpu_relax() asm volatile("pause\n" : : : "memory")
#elif defined(__arm__)
#define cpu_relax() asm volatile("" : : : "memory")
#else
#error Cpu relax not defined for architecture
#endif

#if defined(__x86_64__)

/** 
 * Flush cache for a given area. Calls CLFLUSH on each cacheline
 * touching the region.
 * 
 * @param p Starting point
 * @param size Size to flush in bytes
 */
INLINE void clflush_area(void* p, size_t size)
{
  addr_t paddr = addr_t(p);
  addr_t start_cl = paddr >> CACHE_LINE_SHIFT;
  addr_t end_cl = (paddr+size-1) >> CACHE_LINE_SHIFT;
 
  while(start_cl <= end_cl) {
    /* CLFLUSH doesn't need a fence. CLFLUSHOPT will. need fix for Skylake */
    const void * clp = reinterpret_cast<const void *>(start_cl << CACHE_LINE_SHIFT);
    //    PLOG("flushing cache line: %p", clp);
    __builtin_ia32_clflush(clp);
    start_cl++;
  }  
}

#ifndef likely
#define likely(x)    __builtin_expect (!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x)  __builtin_expect (!!(x), 0)
#endif

#else

#endif

#define check_zero(call) { int rc; \
    if((rc = call) != 0) throw General_exception("%s:%d :call return code (%d) unexpected", __FILE__, __LINE__, rc); }

#define check_nonzero(call) { int rc; \
    if((rc = call) == 0) throw General_exception("%s:%d :call return code (%d) unexpected", __FILE__, __LINE__, rc); }


#endif  // __KIVATI_UTILS_H__
