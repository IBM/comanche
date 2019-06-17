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

#include <common/rand.h>
#include <boost/smart_ptr/detail/spinlock.hpp>
#include <mutex>
#include <thread>
#include "bitmap_ikv.h"
namespace block_alloc_ikv
{
static boost::detail::spinlock _spinlock;
//#define ENABLE_TIMING

/*
 * bitmap region operation support
 *
 * @param bitmap bitmap to operate on
 * @param pos the posion of the bit
 * @param order order of the region
 * @param operation type
 */
int bitmap_ikv::_reg_op(unsigned int pos, unsigned order, reg_op_t reg_op)
{
  int           nbits_reg;   /* number of bits in region */
  int           index;       /* index first long of region in bitmap */
  int           offset;      /* bit offset region in bitmap[index] */
  int           nlongs_reg;  /* num longs spanned by region in bitmap */
  int           nbitsinlong; /* num bits of region in each spanned long */
  unsigned long mask;        /* bitmask for one long of region */
  int           i;           /* scans bitmap by longs */
  int           ret = 0;     /* return value */

#ifdef ENABLE_TIMING
  uint64_t _start; /*for timing*/
#endif

  /*
   * Either nlongs_reg == 1 (for small orders that fit in one long)
   * or (offset == 0 && mask == ~0UL) (for larger multiword orders.)
   */
  nbits_reg   = 1 << order;
  index       = pos / BITS_PER_LONG;
  offset      = pos - (index * BITS_PER_LONG);
  nlongs_reg  = BITS_TO_LONGS(nbits_reg);
  nbitsinlong = nbits_reg < BITS_PER_LONG ? nbits_reg : BITS_PER_LONG;

  /*
   * Can't do "mask = (1UL << nbitsinlong) - 1", as that
   * overflows if nbitsinlong == BITS_PER_LONG.
   */
  mask = (1UL << (nbitsinlong - 1));
  mask += mask - 1;
  mask <<= offset;  // 1111...[offset]

  /*copy the spanned longs*/
#ifdef ENABLE_TIMING
  _start = rdtsc();
#endif
  word_t *tab_start;

  switch (reg_op) {
    case REG_OP_ISFREE: {
      tab_start = _bitdata;
      for (i = 0; i < nlongs_reg; i++) {
        if (tab_start[index + i] & mask) goto done;
      }

      ret = 1; /* all bits in region free (zero) */
      break;
    }

    case REG_OP_ALLOC:
      tab_start = _bitdata + index;

      for (i = 0; i < nlongs_reg; i++) tab_start[i] |= mask;
      break;

    case REG_OP_RELEASE:
      tab_start = _bitdata + index;
      for (i = 0; i < nlongs_reg; i++) tab_start[i] &= ~mask;
      break;
    done:;
  }

#ifdef ENABLE_TIMING
  PDBG("\t[%s]: op: %d: time %lu", __func__, reg_op, rdtsc() - _start);
#endif

  return ret;
}

int bitmap_ikv::find_free_region(unsigned int order)
{
  uint64_t pos, offset;
  size_t   nbits = _capacity;  // total bits in the bitmap

  unsigned int step  = (1U << order);  // size of a tab
  unsigned int ntabs = nbits / step;

#ifdef ENABLE_TIMING
  uint64_t _start, _end_search, _end_set;
  uint64_t cycle_search, cycle_set;
  _start = rdtsc();
#endif
  PDBG(" step = %u, ntabs = %u", step, ntabs);

  uint64_t pos_origin = (genrand64_int64() % ntabs) *
                        step;  // the origin of the scan, star from a random tab

  /* TODO: this mutex can be avoid by manipulating the tab pos to avoid
   * contention!*/
  {
    for (offset = 0; offset + step <= nbits;
         offset +=
         step) {  // go to right most and then start from the left most

      // PINF("    [%s]: end is %d, try to search region order %d from pos
      // %u",__func__, end,  order, pos );
      pos = (offset + pos_origin) % nbits;

      if (!_reg_op(pos, order, REG_OP_ISFREE)) {
        PDBG("thread(%lu) reg not free at %lu",
             std::hash<std::thread::id>{}(std::this_thread::get_id()), pos);
        continue;
      }

#ifdef ENABLE_TIMING
      _end_search = rdtsc();
#endif
      {
        // std::lock_guard<std::mutex> guard(_mutex);
        std::lock_guard<boost::detail::spinlock> guard(_spinlock);
        if (!_reg_op(pos, order, REG_OP_ISFREE)) {
          PDBG("(%lu) retest: reg not free at %lu",
               std::hash<std::thread::id>{}(std::this_thread::get_id()), pos);
          continue;
        }
        _reg_op(pos, order, REG_OP_ALLOC);
        PDBG("thread (%lu) find available slot at %lu",
             std::hash<std::thread::id>{}(std::this_thread::get_id()), pos);
      }

#ifdef ENABLE_TIMING
      _end_set     = rdtsc();
      cycle_search = _end_search - _start;
      cycle_set    = _end_set - _end_search;
      PINF("[%s]: cycle used: search region: %.5lu, set used bits %.5lu",
           __func__, cycle_search, cycle_set);
#endif
      return pos;
    }
    return -1;
  }
}

int bitmap_ikv::release_region(unsigned int pos, unsigned order)
{
  if (_reg_op(pos, order, REG_OP_RELEASE)) {
    throw(General_exception("region release failed"));
  }
  return S_OK;
}

}  // namespace block_alloc_ikv
