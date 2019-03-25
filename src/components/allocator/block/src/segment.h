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
#ifndef __SEGMENT_H__
#define __SEGMENT_H__

#include <api/components.h>
#include <api/block_allocator_itf.h>

using namespace Component;

/** 
 * TODO: 
 * - sort and coalesce
 * - bring slot tail back in.
 * 
 */

class Segment
{
public:

private:
  static constexpr bool option_DEBUG = false;
  
  static constexpr uint32_t MAGIC = 0xABA1EEEE;
  static constexpr uint64_t MSB_64 = (1ULL << 31);
  struct Range
  {
    unsigned free : 1;
    unsigned long lba  : 63; /*< set to zero to indicate free slot */
    uint32_t lb_count;
  } __attribute__((packed));

  static_assert(sizeof(Range)==12,"Range invalid size");
  struct Segment_page
  {
    Segment *         adjacent; /*< to be used to merge across segment boundaries etc. */
    uint32_t          magic;
    uint64_t          lba_begin; /* inclusive */
    uint64_t          lb_count; 
    uint16_t          num_slots; /*< number of ranges in page */
    uint16_t          slot_tail; /*< number of ranges in page */
    Range             ranges[0];
  };
  
public:

  Segment(void * page,
          size_t page_size,
          uint64_t lba_begin,
          uint32_t lb_count,
          bool initialize)
  {
    assert(page);
    assert(page_size > sizeof(Segment_page));
    
    _page = static_cast<Segment_page*>(page);
    
    if(initialize) {      
      _page->magic = MAGIC;
      _page->lba_begin = lba_begin;
      _page->lb_count = lb_count;
      _page->num_slots = (page_size - sizeof(Segment_page)) / sizeof(Range);
      memset(&_page->ranges, 0, page_size - sizeof(Segment_page));
      _page->ranges[0].free = true;
      _page->ranges[0].lba = lba_begin & ~MSB_64;
      _page->ranges[0].lb_count = lb_count;
      _page->slot_tail = 1;
    }
    else {
      assert(_page->magic == MAGIC);
      assert(_page->slot_tail <= _page->num_slots);
    }
  }

  ~Segment() {
  }

  static size_t slot_count(size_t memory_size) {
    return (memory_size - sizeof(Segment_page)) / sizeof(Range);
  }

  void set_adjacent(Segment * s) {
    std::lock_guard<std::mutex> g(_lock);
    _page->adjacent = s;
  }

  Segment * adjacent() const {
    return _page->adjacent;
  }

  void dump_info() {
    std::lock_guard<std::mutex> g(_lock);
    PINF("Segment range:%lu-%lu [%lu blocks] range-slots=%u ",
           _page->lba_begin, _page->lba_begin + _page->lb_count - 1,
         _page->lb_count,
           _page->num_slots);
    for(unsigned i=0;i<_page->slot_tail;i++) {
      Range * r = &_page->ranges[i];
      PINF("\t(%s: %lu-%lu) %u blocks", r->free ? "free" : "used", r->lba, r->lba+r->lb_count-1, r->lb_count);
    }

  }

  status_t alloc(size_t count, lba_t& out_lba) {
    std::lock_guard<std::mutex> g(_lock);
    
    Range * lr = scan_largest_free();
    if(!lr) {
      dump_info();
      throw General_exception("block allocation failed");
    }

    if(lr->lb_count < count)
      return E_EMPTY;
    
    Range * nr = allocate_from(lr, count);
    assert(nr);
    if(option_DEBUG)
      PLOG("returning LBA=%ld from segment (%p)", nr->lba, this);
    out_lba = nr->lba;
    return S_OK;
  }

  bool free(lba_t lba) {
    std::lock_guard<std::mutex> g(_lock);
    Range * r = find(lba);
    if(r==nullptr)
      return false;
    r->free = true;
    return true;
  }

  size_t max_allocation() {
    std::lock_guard<std::mutex> g(_lock);
    Range * r = scan_largest_free();
    if(!r) return 0; // non free
    return r->lb_count;
  }
  
private:
  inline size_t num_slots() {
    std::lock_guard<std::mutex> g(_lock);
    return _page->num_slots;
  }

  inline size_t slot_tail() {
    std::lock_guard<std::mutex> g(_lock);
    return _page->slot_tail;
  }
  

  Range * scan_largest_free()
  {
    Range * p_largest = nullptr;
    uint16_t size_of_largest = 0;
    
    for(unsigned i=0;i<_page->slot_tail;i++) {
      Range * r = &_page->ranges[i];
      if(r->free) {
        if(r->lb_count > size_of_largest) {
          size_of_largest = r->lb_count;
          p_largest = r;
        }
      }
    }
    return p_largest; // could be nullptr
  }

  Range * find(lba_t lba)
  {
    for(unsigned i=0;i<_page->slot_tail;i++) {
      Range * r = &_page->ranges[i];
      if(r->lba == lba)
        return r;
      }
    return nullptr;
  }

  Range * allocate_from(Range * range, size_t lb_count)
  {
    assert(range->free);
    assert(lb_count <= range->lb_count);

    /* exact fit */
    if(lb_count == range->lb_count) {
      range->free = false;
      return range;
    }

    if(_page->num_slots == _page->slot_tail) { /* out of slots */
      throw General_exception("segment slots exhausted; chaining not implemented");
    }

    /* split needed, cut front */
    Range * new_entry = next_free_slot();
    new_entry->free = false;
    new_entry->lb_count = lb_count;
    new_entry->lba = range->lba;

    range->lb_count -= lb_count;
    range->lba += lb_count;
    
    return new_entry;
  }

  Range * next_free_slot()
  {
    if(_page->num_slots == _page->slot_tail)
      return nullptr;

    /* search for free slot up to slot_tail */
    for(unsigned i=0;i<_page->slot_tail;i++) {
      if(_page->ranges[i].lb_count == 0) {
        return &_page->ranges[i];
      }
    }
    Range * r = &_page->ranges[_page->slot_tail];
    _page->slot_tail++;
    return r;
  }
    

  /** 
   * TODO: sort and coalesce
   * 
   */
  void coalesce()
  {
  }

public:
  Segment_page * _page;
  std::mutex     _lock;
  
};

#endif
