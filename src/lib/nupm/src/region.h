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

#ifndef __NUPM_REGION_H__
#define __NUPM_REGION_H__

#include "mappers.h"
#include "rc_alloc_avl.h"
#include <common/exceptions.h> /* API_exception, Logic_exception */
#include <common/utils.h> /* check_aligned */
#include <tbb/cache_aligned_allocator.h>
#include <tbb/scalable_allocator.h>
#include <algorithm>
#include <cassert>
#include <cstddef> /* size_t */
#include <forward_list>
#include <functional>
#include <iostream> /* cout */
#include <list>
#include <map>
#include <memory>
#include <set>
#include <sstream> /* stringstream */

#define SANITY_CHECK 0
namespace nupm
{

/** 
 * Class to manage individual regions on the heap
 * 
 */
class Region {
  static constexpr unsigned _debug_level = 0;

  using list_t = std::forward_list<void *, tbb::scalable_allocator<void *>>;
  using set_t  = std::set<void *, std::less<void*>, tbb::scalable_allocator<void *>>;

  void mark_used(void *p)
  {
#if SANITY_CHECK
    _used.push_front(p);
#else
    ++_use_count;
#endif
  }

  void mark_unused(void *p)
  {
    _free_list.push_front(p);
    --_use_count;
  }

public:
  Region(void *region_ptr, const size_t region_size, const size_t object_size)
    : _object_size(object_size)
    , _base(region_ptr)
    , _top(static_cast<char *>(_base) + region_size)
    , _use_count(0)
  {
    if (_debug_level > 1)
      PLOG("new region: region_base=%p region_size=%lu objsize=%lu capacity=%lu",
           region_ptr, region_size, object_size, region_size / object_size);

    if (region_size % object_size)
      throw std::invalid_argument(
          "Region: objects must fit exactly into region size");

    if (object_size < 8)
      throw std::invalid_argument("Region: minimum object size is 8 bytes");

    byte *     rp    = static_cast<byte *>(region_ptr);
    const auto count = region_size / object_size;

    /* add to free map - all slots are free initially */
    for (size_t i = 0; i < count; i++) {
      _free_set.insert(rp);
      rp += object_size;
    }
  }

  size_t object_size() const
  {
    return _object_size;
  }

  inline bool in_range(void *p) const
  {
    return (_base <= p && p < _top);
  }

  void *allocate()
  {
    /* migrate any remaining entries on map to list */
    if(!_free_set.empty()) {
      for(const auto& i: _free_set) {
        _free_list.push_front(i);
      }
      _free_set.clear();
    }

    if (_free_list.empty()) return nullptr;
    void *p = _free_list.front();
    _free_list.pop_front();
    mark_used(p);
    assert(check_aligned(p, _object_size));
    return p;
  }

  bool free(void *p)
  {
#if SANITY_CHECK
    auto i = _used.begin();

    /* Note: considerable code here to allow implementation of the free list
     * as a std::forward_list rather than a std::list.
     */
    if (*i == p) {
      _used.pop_front();
      mark_unused(p);
      return true;
    }

    auto last = i;
    ++i;
    const auto e = _used.end();
    while (i != e) {
      if (*i == p) {
        _used.erase_after(last);
        mark_unused(p);
        /* TODO: we could check for total free region */
        return true;
      }
      last = i;
      ++i;
    }
    return false;
#else
    mark_unused(p);
    return true;
#endif
  }

  /** 
   * Allocate at is used during the reconstitution phase. To avoid
   * linear scans, we use a map instead of list here.  Elements
   * are then migrated back to the list on free.
   * 
   */
  bool allocate_at(void *ptr)
  {
    if(!_free_set.empty()) { /* after construction, free elements are put on map first */
      if(_free_set.erase(ptr) != 1)
        throw Logic_exception("allocate_at failed to remove from map");
      return true;
    }

    /* otherwise use the list */
    auto i = _free_list.begin();

    if (*i == ptr) {
      _free_list.pop_front();
      mark_used(ptr);
      return true;
    }
    auto last = i;
    i++;

    while (i != _free_list.end()) {
      if (*i == ptr) {
        _free_list.erase_after(last);
        mark_used(ptr);
        return true;
      }
      last = i;
      i++;
    }
    return true;
  }

  std::size_t use_count() const
  {
    return
#if SANITY_CHECK
      _used.size()
#else
      _use_count
#endif
      ;
  }

  void *base() const { return _base; }
  std::size_t size() const { return static_cast<char *>(_top) - static_cast<char *>(_base); }

  void debug_dump(std::string *out_log = nullptr)
  {
    std::stringstream ss;

#if SANITY_CHECK
    for(auto i: _used)
      ss << "u(" << i << ")\n";
#endif
    for(auto i: _free_list)
      ss << "f(" << i << ")\n";
    /* dump the map, using reverse order in an attempt to match
     * the order in which elements will appear in the equivalent list.
     * Should this fail in the future (in the sense that the LB
     * reconstitute test fails, dump a sorted copy of the _free_list,
     * and the free_set.
     */
    for ( auto it = _free_set.rbegin(); it != _free_set.rend(); ++it )
    {
      ss << "f(" << *it << ")\n";
    }
    ss << "\n";
    if(out_log)
      out_log->append(ss.str());
    else
      std::cout << ss.str() << std::endl;
  }

 private:

  const size_t _object_size;
  void * const _base;
  void * const _top;
  list_t       _free_list;
  set_t        _free_set;

#if SANITY_CHECK
  list_t       _used; /* we could do without this, but it guards against misuse */
#else
  std::size_t  _use_count;
#endif
};

/**
 * Region-based heap allocator.  Uses 2^n sized bucket strategy.
 */
class Region_map {
  static constexpr unsigned NUM_BUCKETS    = 64;
  static constexpr int      MAX_NUMA_ZONES = 2;
  static constexpr unsigned _debug_level   = 0;

 public:
  Region_map() {}

  ~Region_map() {}

  void add_arena(void *arena_base, size_t arena_length, int numa_node)
  {
    if (numa_node < 0 || numa_node >= MAX_NUMA_ZONES)
      throw std::invalid_argument("numa node outside max range");

    _arena_allocator.add_managed_region(arena_base, arena_length, numa_node);
  }

  void *allocate(size_t size, int numa_node, size_t alignment)
  {
    if(alignment > 0) {
      if (alignment > size || size % alignment)
        throw std::invalid_argument("alignment should be integral of size and less than size");
    }

    if (unlikely(numa_node < 0 || numa_node >= MAX_NUMA_ZONES))
      throw std::invalid_argument("numa node outside max range");

    void *p = nullptr;

    if (_mapper.could_exist_in_region(size))
    {
      p = allocate_from_existing_region(size, numa_node);
    }

    if (!p)
    {
      /* Two separate conditions lead here:
       *  1. ! _mapper.could_exist_in_region
       *     (did not try to allocate)
       *  2. _mapper.could_exist_in_region && ! allocate_from_existing_region
       *     (tried to allocate, no space available)
       */
      p = allocate_from_new_region(size, numa_node);
    }

    if (!p) throw std::bad_alloc();

    /* debugging */
    if(_debug_level > 2) {
      if(size <= nupm::Large_and_small_bucket_mapper::L0_MAX_SMALL_OBJECT_SIZE) {
        assert(((addr_t)p) % _mapper.rounded_up_object_size(size) == 0); /* small objects should be size-aligned */
      }
    }

    return p;
  }

  void free(void *p, int numa_node, size_t object_size)
  {
    if (unlikely(numa_node < 0 || numa_node >= MAX_NUMA_ZONES))
      throw std::invalid_argument("numa node outside max range");

    /* begin and end of buckets to search */
    const auto start_bucket =
      object_size > 0
      ? _mapper.bucket(object_size)
      : 0
      ;
    const auto end_bucket =
      object_size > 0
      ? start_bucket + 1
      : NUM_BUCKETS
      ;

    auto &node_buckets = _buckets[numa_node];

    if (start_bucket >= NUM_BUCKETS)
      throw std::out_of_range("object size beyond available buckets");

    /* search starting at first eligible bucket, ending at ineligible bucket */
    for (auto i = start_bucket; i != end_bucket; ++i) {
      /* search regions */
      auto &buckets = node_buckets[i];
      auto it =
        std::find_if(
            std::begin(buckets)
            , std::end(buckets)
            , [&p] (auto &r) { return r->in_range(p); }
        );
      if ( it != std::end(buckets) )
      {
        if ( ! (*it)->free(p) )
        {
          throw Logic_exception("region in range, but not free");
        }

        /*
         * bucket *it has one less allocated object.
         * If the bucket has no allocated objects, return it to the arena
         * from whence it came.
         */
        if ((*it)->use_count() == 0)
        {
          auto r = std::move(*it);
          buckets.erase(it);
          _arena_allocator.free(r->base(), numa_node, r->size());
        }

        return;
      }
    }
    throw API_exception("invalid pointer to free (ptr=%p,numa=%d,size=%lu)", p,
                        numa_node, object_size);
  }

  /**
   * Inject a prior allocation.  Marks the memory as allocated.
   *
   * @param      ptr        The pointer
   * @param[in]  size       The size
   * @param[in]  numa_node  The numa node
   */
  void inject_allocation(void *ptr, size_t size, int numa_node)
  {
    assert(ptr);
    assert(size > 0);
    if (unlikely(numa_node < 0 || numa_node >= MAX_NUMA_ZONES))
      throw std::invalid_argument("numa node outside max range");

    auto &node_buckets = _buckets[numa_node];

    /* debugging/ */
    if(_debug_level > 2) {
      if(size <= nupm::Large_and_small_bucket_mapper::L0_MAX_SMALL_OBJECT_SIZE) {
        /* small objects should be aligned with bucket object size */
        assert(((addr_t)ptr) % _mapper.rounded_up_object_size(size) == 0);
      }
    }

    auto bucket = _mapper.bucket(size);

    /* check existing regions in the bucket */
    for (auto &region : node_buckets[bucket]) {
      if (region->in_range(ptr)) {
        if(region->allocate_at(ptr)) return;
        else {
          region->debug_dump();
          PLOG("rounded up object size: %lu", _mapper.rounded_up_object_size(size));
          PLOG("region object size: %lu", region->object_size());
          throw Logic_exception("inject allocation found region, but allocate_at failed (%p,%lu)",
                                ptr, size);
        }
      }
    }

    /* otherwise we have to create the region at the correct position  */
    auto region_size = _mapper.region_size(size);
    assert(region_size > 0);
    auto region_base = _mapper.base(ptr, size);

    if(_debug_level > 2) {
      PLOG("derived (ptr=%p size=%lu) region_base=%p region_size=%lu\n",
           ptr, size, region_base, region_size);
    }

    _arena_allocator.inject_allocation(region_base, region_size, numa_node);

    size_t region_object_size = region_size == size ?
      region_size : _mapper.rounded_up_object_size(size);

    auto new_region =
      std::make_unique<Region>(region_base, region_size, region_object_size);

    new_region->allocate_at(ptr);
    node_buckets[bucket].push_front(std::move(new_region));
  }

  void debug_dump(std::string *out_log = nullptr)
  {
    for (unsigned numa_node = 0; numa_node < MAX_NUMA_ZONES; numa_node++) {
      for (unsigned i = 0; i < NUM_BUCKETS; i++) {
        /* search regions */
        for (auto &region : _buckets[numa_node][i]) {
          region->debug_dump(out_log);
        }
      }
    }
  }

 private:
  void *allocate_from_existing_region(size_t object_size, int numa_node)
  {
    auto bucket = _mapper.bucket(object_size);
    if (bucket >= NUM_BUCKETS)
      throw std::out_of_range("object size beyond available buckets");

    const auto &node_buckets = _buckets[numa_node];
    for (auto &region : node_buckets[bucket]) {
      void *p = region->allocate();
      if (p != nullptr) return p;
    }
    return nullptr;
  }

  void *allocate_from_new_region(size_t object_size, int numa_node)
  {
    auto bucket = _mapper.bucket(object_size);
    if (bucket >= NUM_BUCKETS)
      throw std::out_of_range("object size beyond available buckets");
    auto region_size = _mapper.region_size(object_size);
    assert(region_size > 0);

    /* allocate region */
    auto region_object_size = _mapper.rounded_up_object_size(object_size);
    auto region_base        = _arena_allocator.alloc(region_size, numa_node, region_size);

    assert(region_base);
    assert(check_aligned(region_base, region_object_size));

    auto new_region =
      std::make_unique<Region>(region_base, region_size, region_object_size);
    void *rp = new_region->allocate();

    assert(rp);
    // PLOG("allocate from new (object_size=%lu, region_object_size=%lu, allocation=%p)",
    //      object_size, region_object_size, rp);
    /* add region to bucket */
    auto &node_buckets = _buckets[numa_node];
    node_buckets[bucket].push_front(std::move(new_region));

    return rp;
  }

  /* Note: unused. Why no object_size and numa_zone arguments? */
  void delete_region(Region *region)
  {
    for (unsigned z = 0; z < MAX_NUMA_ZONES; z++) {
      for (unsigned i = 0; i < NUM_BUCKETS; i++) {
        auto &buckets = _buckets[z][i];
        auto iter =
          std::find_if(
            std::begin(buckets)
            , std::end(buckets)
            , [region] (const auto &x) { return x.get() == region; }
          );
        if ( iter != std::end(_buckets[z][i]) )
        {
          _buckets[z][i].erase(iter);
          return;
        }
      }
    }
    throw std::invalid_argument("delete_region: region not found");
  }

 private:
  Bucket_mapper       _mapper;
  nupm::Rca_AVL       _arena_allocator;
  std::list<std::unique_ptr<Region>> _buckets[MAX_NUMA_ZONES][NUM_BUCKETS];
};

}  // namespace nupm
#endif
