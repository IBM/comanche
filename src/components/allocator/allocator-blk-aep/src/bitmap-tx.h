/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

/*
 * bitmap using pmem transaction api
 *
 * some of the design is referred from linux/bitmap.h
 * also see the /source/arch/c6x/mm/dma-corerent.c
 */

#include <libpmemobj.h>


#ifndef _BM_TX_H_
#define _BM_TX_H_

typedef unsigned long word_t; // bitmap operation is done in word
typedef unsigned long * bitmap_ptr;

POBJ_LAYOUT_BEGIN(bitmap_store);
POBJ_LAYOUT_ROOT(bitmap_store, struct bitmap_tx);
POBJ_LAYOUT_TOID(bitmap_store,  word_t);
POBJ_LAYOUT_END(bitmap_store);


struct bitmap_tx{
  uint64_t nbits; // number of bits in the map
  //uint64_t used_bits;
	TOID(word_t) bitdata; // bitmap are stores in words
};

#define BITS_PER_LONG (64)
#define BITS_TO_LONGS(nr) (((nr)+(BITS_PER_LONG-1))/(BITS_PER_LONG))

/*
 * create a bitmap
 *
 * @param pop persist object pool
 * @param bitmap bitmap to operate on
 * @param nbits number of bits in this map
 */
int bitmap_tx_create(PMEMobjpool *pop, TOID(struct bitmap_tx) pbitmap, unsigned  nbits);

/*
 * destroy a bitmap(free all space)
 *
 * @param pop persist object pool
 * @param bitmap bitmap to operate on
 */

int bitmap_tx_destroy(PMEMobjpool *pop, TOID(struct bitmap_tx)  bitmap);


/*
 * zeroing all the map
 *
 * @param pop persist object pool
 * @param bitmap bitmap to operate on
 */

int bitmap_tx_zero(PMEMobjpool *pop,  TOID(struct bitmap_tx) bitmap);

/*
 * find a contiguous aligned mem region
 *
 * @param pop persist object pool
 * @param bitmap the bitmap to operate on 
 * @param order, the order(2^order free blocks) to set from 0 to 1 
 */
int bitmap_tx_find_free_region(PMEMobjpool *pop,  TOID(struct bitmap_tx) bitmap, int order);

/*
 * free a region from bitmap
 *
 * @param pop persist object pool
 * @param bitmap the bitmap to operate on 
 * @param pos the starting bit to reset
 * @param order, the order(2^order bits) to reset
 */
int bitmap_tx_release_region(PMEMobjpool *pop,  TOID(struct bitmap_tx) bitmap, unsigned int pos, int order);

#endif
