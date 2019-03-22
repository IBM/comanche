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

#include <tbb/cache_aligned_allocator.h>
#include <tbb/scalable_allocator.h>
#include <forward_list>
#include <unordered_set>
#include "mappers.h"
#include "rc_alloc_avl.h"

namespace nupm
{

/** 
 * Class to manage individual regions on the heap
 * 
 */
class Region {
  friend class Region_map;

  static constexpr unsigned _debug_level = 0;

  //  using list_t =
  //  std::forward_list<void*,tbb::cache_aligned_allocator<void*>>;
  using list_t = std::forward_list<void *, tbb::scalable_allocator<void *>>;
  // using list_t = std::forward_list<void*>;

 protected:
  Region(void *region_ptr, const size_t region_size, const size_t object_size)
    : _object_size(object_size)
  {
    if (_debug_level > 1)
      PLOG("new region: region_base=%p region_size=%lu objsize=%lu capacity=%lu",
           region_ptr, region_size, object_size, region_size / object_size);

    if (region_size % object_size)
      throw std::invalid_argument(
          "Region: objects must fit exactly into region size");

    if (object_size < 8)
      throw std::invalid_argument("Region: minimum object size is 8 bytes");

    _base = reinterpret_cast<addr_t>(region_ptr);
    _top  = _base + region_size;

    byte *     rp    = static_cast<byte *>(region_ptr);
    const auto count = region_size / object_size;
    for (size_t i = 0; i < count; i++) {
      _free.push_front(rp);
      rp += object_size;
    }
  }

  size_t object_size() const
  {
    return _object_size;
  }
  
  inline bool in_range(void *p)
  {
    auto addr = reinterpret_cast<addr_t>(p);
    return (addr >= _base && addr < _top);
  }

  void *allocate()
  {
    if (_free.empty()) return nullptr;
    void *p = _free.front();
    _free.pop_front();
    _used.push_front(p);
    assert(check_aligned(p, _object_size));
    return p;
  }

  bool free(void *p)
  {
    auto i = _used.begin();
    if (*i == p) {
      _used.pop_front();
      _free.push_front(p);
      return true;
    }

    auto last = i;
    i++;
    while (i != _used.end()) {
      if (*i == p) {
        _used.erase_after(last);
        _free.push_front(p);
        /* TODO: we could check for total free region */
        return true;
      }
      last = i;
      i++;
    }
    return false;
  }

  bool allocate_at(void *ptr)
  {
    auto i = _free.begin();
    if (*i == ptr) {
      _free.pop_front();
      _used.push_front(ptr);
      return true;
    }
    auto last = i;
    i++;
    while (i != _free.end()) {
      if (*i == ptr) {
        _free.erase_after(last);
        _used.push_front(ptr);
        return true;
      }
      last = i;
      i++;
    }
    return false;
  }

  void debug_dump(std::string *out_log = nullptr)
  {
    std::stringstream ss;

    for(auto i: _used)
      ss << "u(" << i << ")\n";

    for(auto i: _free)
      ss << "f(" << i << ")\n";
    ss << "\n";
    if(out_log)
      out_log->append(ss.str());
    else
      std::cout << ss.str() << std::endl;  
  }
  
 private:
  const size_t _object_size;
  addr_t _base, _top;
  list_t _free;
  list_t _used; /* we could do with this, but it guards against misuse */
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

    if(_mapper.could_exist_in_region(size))
      p = allocate_from_existing_region(size, numa_node);
    
    if (!p)
      p = allocate_from_new_region(size, numa_node);

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

    if (object_size > 0) {
      auto bucket = _mapper.bucket(object_size);
      if (bucket >= NUM_BUCKETS)
        throw std::out_of_range("object size beyond available buckets");
      /* search regions */
      for (auto &region : _buckets[numa_node][bucket]) {
        if (region->in_range(p)) {
          if (region->free(p))
            return;
          else
            throw Logic_exception("region in range, but not free");
        }
      }
    }
    else {
      /* search all the buckets */
      for (unsigned i = 0; i < NUM_BUCKETS; i++) {
        /* search regions */
        for (auto &region : _buckets[numa_node][i]) {
          if (region->in_range(p)) {
            if (region->free(p))
              return;
            else
              throw Logic_exception("region in range, but not free");
          }
        }
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

    /* debugging */
    if(_debug_level > 2) {
      if(size <= nupm::Large_and_small_bucket_mapper::L0_MAX_SMALL_OBJECT_SIZE) {
        /* small objects should be aligned with bucket object size */
        assert(((addr_t)ptr) % _mapper.rounded_up_object_size(size) == 0); 
      }
    }

    auto bucket = _mapper.bucket(size);

    /* check existing regions in the bucket */
    for (auto &region : _buckets[numa_node][bucket]) {
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

    Region *new_region =
        new Region(region_base, region_size, region_object_size);
    
    _buckets[numa_node][bucket].push_front(new_region);
    new_region->allocate_at(ptr);
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

    for (auto &region : _buckets[numa_node][bucket]) {
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

    Region *new_region = new Region(region_base, region_size, region_object_size);
    void *rp = new_region->allocate();
    
    assert(rp);
    // PLOG("allocate from new (object_size=%lu, region_object_size=%lu, allocation=%p)",
    //      object_size, region_object_size, rp);
    /* add region to bucket */
    _buckets[numa_node][bucket].push_front(new_region);

    return rp;
  }

  void delete_region(Region *region)
  {
    for (unsigned z = 0; z < MAX_NUMA_ZONES; z++) {
      for (unsigned i = 0; i < NUM_BUCKETS; i++) {
        auto iter = _buckets[z][i].begin();
        do {
          if (*iter == region) {
            _buckets[z][i].erase(iter);
            return;
          }
          iter++;
        } while (iter != _buckets[z][i].end());
      }
    }
    throw std::invalid_argument("delete_region: region not found");
  }

 private:
  Bucket_mapper       _mapper;
  nupm::Rca_AVL       _arena_allocator;
  std::list<Region *> _buckets[MAX_NUMA_ZONES][NUM_BUCKETS];
};

}  // namespace nupm
#endif
