#ifndef __BLOCK_BITMAP_H__
#define __BLOCK_BITMAP_H__

#include <common/cycles.h>
#include <common/logging.h>
#include <common/rand.h>
#include <core/avl_malloc.h>
#include <core/slab.h>
#include <limits.h>
#include <stdint.h>

/**
 * Element tracking class based on bitmap. This class is
 * thread-safe (after construction) and uses fine-grained
 * compare-and-exchange operations to modify the bitmap
 *
 */
class Block_bitmap {
 public:
  static constexpr size_t ELEMENTS_PER_BITMAP_PAGE = (510 * 64);
  static constexpr size_t QWORDS_PER_PAGE          = 510;

 private:
  static constexpr bool option_DEBUG = true;

  // 4K structure - 32K pages per Bitmap_page
  typedef struct {
    int32_t  _free_count;            /* number of free pages: bloom filter */
    int32_t  _chain_id;              /* number of free pages: bloom filter */
    off64_t  _next_lba;              /* lba of linked region bitmap */
    uint64_t _bits[QWORDS_PER_PAGE]; /* use fine grained locking */
  } __attribute__((packed)) Bitmap_page;

 public:
  /**
   * Constructor based on existing allocated memory (e.g., persistent).
   * Caller is responsible for deallocation of memory.
   *
   * @param buffer Pointer to pre-allocated buffer
   * @param buffer_len Size of pre-allocated buffer in bytes
   * @param capacity Capacity of bitmap in blocks
   * @param initialize Initialization flag
   */
  Block_bitmap(void *   buffer,
               size_t   buffer_len,
               uint64_t capacity,
               bool     initialize = true)
      : _capacity(capacity)
  {
    if (buffer == nullptr) throw API_exception("bad argument");

    _num_bitmap_pages = (capacity / ELEMENTS_PER_BITMAP_PAGE) + 1;
    if (_num_bitmap_pages * sizeof(Bitmap_page) > buffer_len)
      throw API_exception(
          "Block_bitmap: insufficient memory space for capacity");

    if (option_DEBUG)
      PLOG("Block_bitmap: allocated %ld pages (%ld MB)", _num_bitmap_pages,
           REDUCE_MB(_num_bitmap_pages * 4096));

    _bitmaps = static_cast<Bitmap_page *>(buffer);

    if (initialize) {
      memset(_bitmaps, 0, sizeof(Bitmap_page) * _num_bitmap_pages);
      for (unsigned p = 0; p < _num_bitmap_pages; p++) {
        _bitmaps[p]._free_count = ELEMENTS_PER_BITMAP_PAGE;
        if (p < (_num_bitmap_pages - 1))
          _bitmaps[p]._chain_id = p;
        else
          _bitmaps[p]._chain_id = -1;
      }
    }

    /* TODO: trim bitmaps to size */
  }

  /**
   * Allocate a block, at a given hint address if possible
   *
   * @param hint
   *
   * @return
   */
  int64_t alloc_block(uint64_t hint)
  {
    unsigned p, i, b;
    get_coordinates(hint, p, i, b);

    if ((hint > 0) && alloc_block_at(p, i, b)) {
      decrement_free_count(p);
      return hint;
    }

    /* for the moment, simple first fit policy */
    /* this could be greatly optimized         */
    for (p = 0; p < _num_bitmap_pages; p++) {
      if (_bitmaps[p]._free_count == 0) continue; /* ok, not safe */

      for (i = 0; i < QWORDS_PER_PAGE; i++) {
        if (_bitmaps[p]._bits[i] == ULLONG_MAX) continue;

        for (b = 0; b < 64; b++) {
          if (alloc_block_at(p, i, b)) {
            decrement_free_count(p);
            int64_t allocated_block =
                (int64_t)((p * ELEMENTS_PER_BITMAP_PAGE) + (i * 64) + b);
            return allocated_block;
          }
        }
      }
    }
    throw API_exception("out of blocks");
    return -1;
  }

  /**
   * Mark a block as free
   *
   * @param block
   */
  void free_block(uint64_t block)
  {
    unsigned p, i, b;
    get_coordinates(block, p, i, b);

    unsigned retries = 8;
  retry:
    uint64_t old_qword = _bitmaps[p]._bits[i];
    if (!(old_qword & (1UL << b))) throw API_exception("block bit not set");

    uint64_t new_qword = _bitmaps[p]._bits[i] & ~(1UL << b);
    if (__sync_bool_compare_and_swap(&_bitmaps[p]._bits[i], old_qword,
                                     new_qword)) {
      increment_free_count(p);
      return;
    }
    if (retries > 0) {
      retries--;
      goto retry;
    }
    throw API_exception("unable to clear block bit");
  }

  /**
   * Allocate a contigious chunk of 64 blocks
   *
   * @param hint
   *
   * @return
   */
  int64_t alloc_64_blocks(uint64_t hint)
  {
    if (hint % 64)
      throw API_exception("hint for 64 block allocation must be 64 aligned");

    unsigned p, i, b;
    get_coordinates(hint, p, i, b);

    if (alloc_64_blocks_at(p, i)) {
      decrement_free_count(p, 64);
      return hint;
    }

    /* for the moment, simple first fit policy */
    /* this could be greatly optimized         */
    for (p = 0; p < _num_bitmap_pages; p++) {
      if ((_bitmaps[p]._free_count == 0) || (_bitmaps[p]._bits[i] > 0))
        continue; /* ok, not safe */

      for (i = 0; i < QWORDS_PER_PAGE; i++) {
        if (_bitmaps[p]._bits[i] == 0ULL) {
          if (alloc_64_blocks_at(p, i)) {
            decrement_free_count(64);
            return (int64_t)((p * ELEMENTS_PER_BITMAP_PAGE) + (i * 64));
          }
        }
      }
    }
    throw API_exception("out of blocks (alloc_64_blocks)");
    return -1;
  }

  /**
   * Mark a block as free
   *
   * @param block
   */
  void free_64_blocks(uint64_t block)
  {
    if (block % 64) throw API_exception("bad address param to free_64_blocks");

    unsigned p, i, b;
    get_coordinates(block, p, i, b);

    unsigned retries = 8;
  retry:

    if (_bitmaps[p]._bits[i] != ULLONG_MAX)
      throw API_exception("64 block bit set not all set");

    if (__sync_bool_compare_and_swap(&_bitmaps[p]._bits[i], ULLONG_MAX, 0ULL)) {
      increment_free_count(p);
      return;
    }
    if (retries > 0) {
      retries--;
      goto retry;
    }
    throw API_exception("unable to clear block bit");
  }

  /**
   * Determine how much memory to allocate for a given block capacity
   *
   * @param capacity Capacity in blocks
   *
   * @return Number of bytes to allocate
   */
  static size_t size_for(uint64_t capacity)
  {
    return ((capacity / ELEMENTS_PER_BITMAP_PAGE) + 1) * sizeof(Bitmap_page);
  }

 private:
  /**
   * Decrement bloom-type filter ref. count
   *
   * @param p Page
   */
  void decrement_free_count(unsigned p, unsigned dec = 1)
  {
    Bitmap_page *bp = &_bitmaps[p];
    __sync_fetch_and_sub(&bp->_free_count, dec);
    assert(bp->_free_count >= 0);
  }

  /**
   * Increment bloom-type filter ref. count
   *
   * @param p Page
   */
  void increment_free_count(unsigned p, unsigned inc = 1)
  {
    Bitmap_page *bp = &_bitmaps[p];
    __sync_fetch_and_add(&bp->_free_count, inc);
    assert(bp->_free_count > 0);
  }

  /**
   * Allocate a block at a given position (hint) if possible
   *
   * @param hint logical block address hin
   *
   * @return true on success
   */
  bool alloc_block_at(unsigned p, unsigned i, unsigned b)
  {
    unsigned retries = 8;
  retry:
    uint64_t old_qword = _bitmaps[p]._bits[i];
    if (old_qword & (1UL << b)) return false; /* bit already set */
    uint64_t new_qword = _bitmaps[p]._bits[i] | (1UL << b);
    if (__sync_bool_compare_and_swap(&_bitmaps[p]._bits[i], old_qword,
                                     new_qword)) {
      return true;
    }
    if (retries > 0) {
      retries--;
      goto retry;
    }
    else
      return false;
  }

  /**
   * Allocate 64 contiguous blocks
   *
   * @param p Page
   * @param i Index in page array
   *
   * @return
   */
  bool alloc_64_blocks_at(unsigned p, unsigned i)
  {
    unsigned retries = 8;
  retry:

    uint64_t old_qword = _bitmaps[p]._bits[i];
    if (old_qword > 0) return false; /* cannot allocate */

    uint64_t new_qword = ~(0ULL);
    if (__sync_bool_compare_and_swap(&_bitmaps[p]._bits[i], old_qword,
                                     new_qword)) {
      return true;
    }
    if (retries > 0) {
      retries--;
      goto retry;
    }
    else
      return false;
  }

  /**
   * Get coordinates for a given block
   *
   * @param lea Logical element address
   * @param page Bitmap page (from 0)
   * @param index Index to 64bit work in page (from 0)
   * @param bit Bit in 64bit word
   */
  void get_coordinates(uint64_t  lea,
                       unsigned &page,
                       unsigned &index,
                       unsigned &bit)
  {
    uint64_t remainder;
    page      = lea / ELEMENTS_PER_BITMAP_PAGE;
    remainder = lea % ELEMENTS_PER_BITMAP_PAGE;
    index     = remainder / 64;
    bit       = remainder % 64;
  }

 private:
  size_t       _capacity;         /*< defines capacity in blocks */
  size_t       _num_bitmap_pages; /*< number of 4K metadata pages */
  Bitmap_page *_bitmaps;          /* pointer to bitmap page array */
};

#endif
