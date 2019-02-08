#ifndef __NUPM_REGION_H__
#define __NUPM_REGION_H__

#include <forward_list>
#include <unordered_set>
#include "rc_alloc_avl.h"
#include "mappers.h"
#include <tbb/cache_aligned_allocator.h>
#include <tbb/scalable_allocator.h>

namespace nupm
{
  
class Region
{
  friend class Region_map;

  static constexpr unsigned _debug_level = 3;
  
  struct ptr_t {
    void * ptr;
  } __attribute__((aligned(8)));
  
  //  using list_t = std::forward_list<ptr_t,tbb::cache_aligned_allocator<ptr_t>>;
  using list_t = std::forward_list<ptr_t,tbb::scalable_allocator<ptr_t>>;
  //  using list_t = std::forward_list<ptr_t>;
protected:
  Region(void * region_ptr,
         const size_t region_size,
         const size_t obj_size)
  {
    if(_debug_level > 1)
      PLOG("Region ctor: region_base=%p region_size=%lu objsize=%lu objcap=%lu",
           region_ptr, region_size, obj_size, region_size / obj_size);
    
    if(region_size % obj_size)
      throw std::invalid_argument("Region: objects must fit exactly into region size");

    if(obj_size < 8)
      throw std::invalid_argument("Region: minimum object size is 8 bytes");
    
    char * rp = static_cast<char*>(region_ptr);
    const auto count = region_size / obj_size;
    for(size_t i=0;i<count; i++) {
      _free.push_front({rp});
      rp+=obj_size;
    }
  }

  void * allocate() {
    if(_free.empty())
      return nullptr;
    void * p = _free.front().ptr;
    _free.pop_front();
    _used.push_front({p});
    return p;
  }

  bool free(void * p) {
    auto i = _used.begin();
    if(i->ptr == p)
      _used.pop_front();

    auto last = i;
    i++;
    while(i != _used.end()) {
      if(i->ptr == p) {
        _used.erase_after(last);
        _free.push_front({p});
        /* TODO: we could check for total free region */
        return true;
      }
      i++;
      last++;
    }
    return false;
  }
    
  list_t _free;
  list_t _used; /* we could do with this, but it guards against misuse */
};

  
class Region_map
{
  static constexpr unsigned NUM_BUCKETS = 64;
  static constexpr int MAX_NUMA_ZONES = 2;
  static constexpr unsigned _debug_level = 3;
  
public:
  Region_map() {
  }

  virtual ~Region_map() {
  }
  
  void add_arena(void * arena_base, size_t arena_length, int numa_node) {
    if(numa_node >= MAX_NUMA_ZONES)
      throw std::invalid_argument("numa node outside max range");
    
    _arena_allocator.add_managed_region(arena_base, arena_length, numa_node);
  }

  void * allocate(size_t size, int numa_node) {
    void * p = allocate_from_existing_region(size, numa_node);
    if(!p)
      p = allocate_from_new_region(size, numa_node);
    if(!p)
      throw std::bad_alloc();
    return p;
  }

  void free(void * p, int numa_node, size_t obj_size) {
    if(numa_node >= MAX_NUMA_ZONES)
      throw std::invalid_argument("numa node outside max range");

    if(obj_size > 0) {
      auto bucket = _mapper.bucket(obj_size);
      if(bucket >= NUM_BUCKETS)
        throw std::out_of_range("object size beyond available buckets");
      /* search regions */
      for(auto& region : _buckets[numa_node][bucket]) {
        if(region->free(p))
          return;
      }
    }
    else {
      /* search all the buckets */
      for(unsigned i=0;i<NUM_BUCKETS;i++) {
        /* search regions */
        for(auto& region : _buckets[numa_node][i]) {
          if(region->free(p))
            return;
        }
      }
    }
    throw std::invalid_argument("bad pointer to free");
  }

private:

  
  void * allocate_from_existing_region(size_t obj_size, int numa_node) {
    auto bucket = _mapper.bucket(obj_size);
    if(bucket >= NUM_BUCKETS)
      throw std::out_of_range("object size beyond available buckets");
    for(auto& region : _buckets[numa_node][bucket]) {
      void * p = region->allocate();
      if(p)
        return p;
    }
    return nullptr;
  }

  void * allocate_from_new_region(size_t obj_size, int numa_node) {
    auto bucket = _mapper.bucket(obj_size);
    if(bucket >= NUM_BUCKETS)
      throw std::out_of_range("object size beyond available buckets");
    auto region_size = _mapper.region_size(obj_size);
    assert(region_size > 0);
    auto region_base = _arena_allocator.alloc(region_size, numa_node, obj_size); /* align by object size */
    assert(region_base);
    auto region_object_size = _mapper.rounded_up_object_size(obj_size);
    Region * new_region = new Region(region_base, region_size, region_object_size);
    void * rp = new_region->allocate();
    attach_region(new_region, obj_size, numa_node);
    return rp;
  }

  void attach_region(Region * region, size_t obj_size, int numa_node) {
    auto bucket = _mapper.bucket(obj_size);
    if(bucket >= NUM_BUCKETS)
      throw std::out_of_range("object size beyond available buckets");
    _buckets[numa_node][bucket].push_front(region);
  };

  void delete_region(Region * region) {
    for(unsigned z=0;z<MAX_NUMA_ZONES;z++) {
      for(unsigned i=0;i<NUM_BUCKETS;i++) {
        auto iter = _buckets[z][i].begin();
        do {
          if(*iter == region) {
            _buckets[z][i].erase(iter);
            return;
          }
          iter++;
        }
        while(iter != _buckets[z][i].end());
      }
    }
    throw std::invalid_argument("delete_region: region not found");
  }

private:
  Log2_bucket_mapper  _mapper;
  nupm::Rca_AVL       _arena_allocator;
  std::list<Region *> _buckets[MAX_NUMA_ZONES][NUM_BUCKETS];
};

} // nupm
#endif
