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
  Authors:

  Copyright (C) 2018, Daniel G. Waddington <daniel.waddington@ibm.com>
*/

#ifndef __COMANCHE_LF_BITMAP_H__
#define __COMANCHE_LF_BITMAP_H__

#include <assert.h>
#include <common/cycles.h>
#include <common/rand.h>
#include <stdint.h>

namespace Core
{
/**
 * Relocatable, lock-free bitmap. Should be used with placement operator.
 *
 * e.g. void * ptr = malloc(4096);
 *      Core::Relocatable_LF_bitmap * slab = new (ptr)
 * Core::Relocatable_LF_bitmap(4096,128);
 */
class Relocatable_LF_bitmap {
  static constexpr bool option_DEBUG = true;

 public:
  using range_type = unsigned;

  Relocatable_LF_bitmap(size_t memory_size, size_t n_elements, bool init = true)
      : _n_elements(n_elements) {
    if (n_elements % 64)
      throw API_exception("n_elements size parameter should be modulo 64");
    if (memory_size < (sizeof(Relocatable_LF_bitmap) + (n_elements / 8)))
      throw API_exception(
          "insufficient memory for requested number of elements");

    if (init) {
      memset(this, 0xFF, memory_size);
      _magic = MAGIC;
      _n_elements = n_elements;
      _n_qwords = ((_n_elements + 63) / 64);
    }
    else {
      if (_magic != MAGIC)
        throw API_exception(
            "reconstruction from memory that doesn't match magic");
    }

    if (option_DEBUG)
      PLOG("Relocatable_LF_bitmap: this=%p,  memory_size=%lu, qwords=%u, "
           "n_elements=%u",
           this, memory_size, _n_qwords, _n_elements);
  }

  /**
   * Allocate a slot
   *
   *
   * @return Slot index counting from 0 (throws exception on full)
   */
  range_type allocate() {
    assert(_n_qwords > 0);
    range_type qword = rand() % _n_qwords;
    range_type curr_qword = qword;
    while (curr_qword != _n_qwords) {
      if (_bitmap[curr_qword] == 0) { /* short-circuit empty condition */
        curr_qword++;
        continue;
      }
      unsigned offset = __builtin_ffsll(_bitmap[curr_qword]);
      if (offset == 0) continue;
      offset--;
      uint64_t old_qword = _bitmap[curr_qword];
      uint64_t new_qword = old_qword & ~(1ULL << offset); /* clear bit */
      if (__sync_bool_compare_and_swap(&_bitmap[curr_qword], old_qword,
                                       new_qword)) {
        return (curr_qword * 64) + offset;
      }
      else {
        continue;
      }
    }
    curr_qword = 0;
    while (curr_qword != qword) {
      if (_bitmap[curr_qword] == 0) { /* short-circuit empty condition */
        curr_qword++;
        continue;
      }
      unsigned offset = __builtin_ffsll(_bitmap[curr_qword]);
      if (offset == 0) continue;
      offset--;
      uint64_t old_qword = _bitmap[curr_qword];
      uint64_t new_qword = old_qword & ~(1ULL << offset); /* clear bit */
      if (__sync_bool_compare_and_swap(&_bitmap[curr_qword], old_qword,
                                       new_qword)) {
        return (curr_qword * 64) + offset;
      }
      else {
        continue;
      }
    }

    throw API_exception("no more bits available");

    return 0;
  }

  /**
   * Free slot
   *
   * @param slot Slot index counting from 0
   */
  void free(range_type slot) {
    range_type bitmap_qword = slot / 64;
    unsigned offset = slot % 64;
    if ((_bitmap[bitmap_qword] & (1ULL << offset)))
      throw API_exception("bad slot: %u qword=%lx", slot,
                          _bitmap[bitmap_qword]);
  retry:
    uint64_t old_qword = _bitmap[bitmap_qword];
    uint64_t new_qword = old_qword | (1ULL << offset); /* set bit */
    if (!__sync_bool_compare_and_swap(&_bitmap[bitmap_qword], old_qword,
                                      new_qword)) {
      goto retry;
    }
  }

  static size_t required_memory_size(size_t n_elements) {
    return sizeof(Relocatable_LF_bitmap) + (8 * ((n_elements + 63) / 64));
  }

  size_t unsafe_free_count() {
    size_t total_free = 0;
    for (unsigned q = 0; q < _n_qwords; q++) {
      total_free += __builtin_popcountll(_bitmap[q]);
    }
    return total_free;
  }

 private:
  INLINE uint64_t rand() {
    return genrand64_int64();

    // unsigned int r;
    // __builtin_ia32_rdrand32_step(&r);

    //    return r;
  }

 private:
  static constexpr uint32_t MAGIC = 0x10101010;

  uint32_t _magic;
  uint32_t _n_elements;
  uint32_t _n_qwords;
  uint32_t _resvd;
  uint64_t _bitmap[0];

} __attribute__((packed));

}  // namespace Core

#endif
