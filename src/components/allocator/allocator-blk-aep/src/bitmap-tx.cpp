/*
 * Bitmap implementation using pmemobj
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "bitmap-tx.h"

#include <boost/smart_ptr/detail/spinlock.hpp>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/cycles.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <common/rand.h>

static boost::detail::spinlock _spinlock;
//static std::mutex _mutex;

//#define ENABLE_TIMING
enum{
  REG_OP_ISFREE,  // region is all zero bits
  REG_OP_ALLOC,   //set all bits in this region
  REG_OP_RELEASE, //clear all bits in region
};

/*
 * bitmap region operation support
 *
 * @param bitmap bitmap to operate on 
 * @param pos the posion of the bit
 * @param order order of the region
 * @param operation type
 */
static int __reg_op(PMEMobjpool *pop,  TOID(struct bitmap_tx) bitmap, unsigned int pos, unsigned order, int reg_op)
{
	int nbits_reg;		/* number of bits in region */
	int index;		/* index first long of region in bitmap */
	int offset;		/* bit offset region in bitmap[index] */
	int nlongs_reg;		/* num longs spanned by region in bitmap */
	int nbitsinlong;	/* num bits of region in each spanned long */
	unsigned long mask;	/* bitmask for one long of region */
	int i;			/* scans bitmap by longs */
	int ret = 0;		/* return value */

#ifdef ENABLE_TIMING
  uint64_t _start; /*for timing*/
#endif

	/*
	 * Either nlongs_reg == 1 (for small orders that fit in one long)
	 * or (offset == 0 && mask == ~0UL) (for larger multiword orders.)
	 */
	nbits_reg = 1 << order;
	index = pos / BITS_PER_LONG;
	offset = pos - (index * BITS_PER_LONG);
	nlongs_reg = BITS_TO_LONGS(nbits_reg);
	nbitsinlong = nbits_reg < BITS_PER_LONG ? nbits_reg: BITS_PER_LONG;


	/*
	 * Can't do "mask = (1UL << nbitsinlong) - 1", as that
	 * overflows if nbitsinlong == BITS_PER_LONG.
	 */
	mask = (1UL << (nbitsinlong - 1));
	mask += mask - 1;
	mask <<= offset; // 1111...[offset]

  TOID(word_t) bitdata = D_RO(bitmap)->bitdata;

  /*copy the spanned longs*/
#ifdef ENABLE_TIMING 
  _start = rdtsc();
#endif

  switch (reg_op) {
  case REG_OP_ISFREE:
    {
      const word_t * tab_start = D_RO(bitdata);
      for (i = 0; i < nlongs_reg; i++) {
        if (tab_start[index + i] & mask)
          goto done;
      }

      ret = 1;	/* all bits in region free (zero) */
      break;
    }

  case REG_OP_ALLOC:
    TX_BEGIN(pop){
      pmemobj_tx_add_range(bitdata.oid, index*sizeof(long), nlongs_reg*sizeof(long) );
      for (i = 0; i < nlongs_reg; i++)
        D_RW(bitdata)[index + i] |= mask;
    }TX_ONABORT{
    PERR("%s: transaction aborted: %s\n", __func__,
      pmemobj_errormsg());
    abort();
    }TX_END
    break;

  case REG_OP_RELEASE:

    TX_BEGIN(pop){
      pmemobj_tx_add_range(bitdata.oid, index*sizeof(long), nlongs_reg*sizeof(long) );
      for (i = 0; i < nlongs_reg; i++)
        D_RW(bitdata)[index + i] &= ~mask;
    }TX_ONABORT{
    PERR("%s: transaction aborted: %s\n", __func__,
      pmemobj_errormsg());
    abort();
    }TX_END

    break;
done:
    ;
  }

#ifdef ENABLE_TIMING
  PDBG("\t[%s]: op: %d: time %lu",__func__, reg_op,   rdtsc()- _start);
#endif

  return ret;
}

int bitmap_tx_zero(PMEMobjpool *pop,  TOID(struct bitmap_tx) bitmap){
  size_t nbits = D_RO(bitmap)->nbits;
  size_t mem_size = BITS_TO_LONGS(nbits)*sizeof(long);

  // TX_MEMSET has the add_range built in
  TX_BEGIN(pop){
    TX_MEMSET(D_RW((D_RW(bitmap)->bitdata)), 0, mem_size);
  }TX_ONABORT{
		PERR("%s: transaction aborted: %s\n", __func__,
			pmemobj_errormsg());
		abort();
  }TX_END
  
  return 0;
}

int bitmap_tx_create(PMEMobjpool *pop, TOID(struct bitmap_tx) bitmap, unsigned nbits){
  // space for all the bits
  size_t sz = BITS_TO_LONGS(nbits)*sizeof(long);

  TX_BEGIN(pop){
    TX_ADD(bitmap); // since we need let the root obj konw where is bitdata

    D_RW(bitmap)->nbits = nbits;
    D_RW(bitmap)->bitdata = TX_ZALLOC(word_t, sz);
  }TX_ONABORT{
		PERR("%s: transaction aborted create sz=%lu bitmap: %s\n",
        __func__, sz,
			pmemobj_errormsg());
		abort();
  }TX_END

  return 0;
}

int bitmap_tx_destroy(PMEMobjpool *pop, TOID(struct bitmap_tx)  bitmap){
  int ret = -1;

  TX_BEGIN(pop){
    TX_ADD(bitmap);
    TX_FREE(D_RW(bitmap)->bitdata);
    D_RW(bitmap)->bitdata = TOID_NULL(word_t);

    // can I free here?
    //TX_FREE(bitmap);
  } TX_ONABORT {
		fprintf(stderr, "transaction aborted: %s\n",
			pmemobj_errormsg());
    return ret;
	} TX_END

  // free root object
  ret = 0;
  return ret;
}


/* 
 * this will scan the bitmap by regions of size order
 */
int bitmap_tx_find_free_region(PMEMobjpool *pop,  TOID(struct bitmap_tx) bitmap, unsigned order){
  uint64_t pos, offset;
  size_t nbits = D_RO(bitmap)->nbits; // total bits in the bitmap


  unsigned int step =(1U << order); // size of a tab
  unsigned int ntabs = nbits/step;

#if ENABLE_TIMING
  uint64_t _start, _end_search, _end_set;
  uint64_t cycle_search, cycle_set;
  _start = rdtsc();
#endif
  PDBG(" step = %u, ntabs = %u", step, ntabs);

  uint64_t pos_origin = (genrand64_int64()%ntabs)*step; // the origin of the scan, star from a random tab

  /* TODO: this mutex can be avoid by manipulating the tab pos to avoid contention!*/
  {
    std::lock_guard<boost::detail::spinlock> guard(_spinlock);
    //std::lock_guard<std::mutex> guard(_mutex);

    for(offset = 0; offset + step <=nbits; offset += step){ // go to right most and then start from the left most

      //PINF("    [%s]: end is %d, try to search region order %d from pos %u",__func__, end,  order, pos );
      pos = (offset + pos_origin)%nbits;

      if(! __reg_op(pop, bitmap, pos, order, REG_OP_ISFREE))
        continue;

#if ENABLE_TIMING
      _end_search = rdtsc();
#endif

      __reg_op(pop, bitmap, pos, order, REG_OP_ALLOC);

#if ENABLE_TIMING
      _end_set = rdtsc();
      cycle_search = _end_search - _start;
      cycle_set = _end_set - _end_search;
      PDBG("[%s]: cycle used: search region: %.5lu, set used bits %.5lu",__func__,  cycle_search, cycle_set);
#endif
      return pos;
    }
    return -1;
  }
}

int bitmap_tx_release_region(PMEMobjpool *pop,  TOID(struct bitmap_tx) bitmap, unsigned int pos, unsigned order){
  if(__reg_op(pop, bitmap, pos, order, REG_OP_RELEASE)){
      throw(General_exception("region release failed"));
      }
  return S_OK;
     
}
