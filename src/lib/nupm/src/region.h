#ifndef __NUPM_REGION_H__
#define __NUPM_REGION_H__

#include "rc_alloc_avl.h"

namespace nupm
{
  
class Region
{
  friend class Region_map;

  using list_t = std::vector<void*>;
protected:
  Region(void * region_ptr, const size_t region_size, const size_t obj_size) {
    if(region_size % obj_size)
      throw std::invalid_argument("Region: objects must fit exactly into region size");

    char * rp = static_cast<char*>(region_ptr);
    for(size_t i=0;i<(region_size / obj_size); i++) {
      _free.push_back(rp);
      rp+=obj_size;
    }
  }

  void * allocate() {
    if(_free.empty())
      return nullptr;
    void * p = _free.back();
    _free.pop_back();
    _used.push_back(p);
  }

  list_t _free;
  list_t _used; /* we could do with this, but it guards against misuse */
};

class Region_map
{
  static constexpr unsigned NUM_BUCKETS = 64;
  static constexpr int MAX_NUMA_ZONES = 2;
  
public:
  Region_map() {
  }
  
  void add_arena(void * arena_base, size_t arena_length, int numa_node) {
    if(numa_node >= MAX_NUMA_ZONES)
      throw std::invalid_argument("numa node outside max range");
    
    _arena_allocator.add_managed_region(arena_base, arena_length, numa_node);
  }

  

private:
  void * allocate_from_existing_region(size_t s, int numa_node) {
    auto bucket = get_log2_bin(s);
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
    auto bucket = get_log2_bin(obj_size);
    if(bucket >= NUM_BUCKETS)
      throw std::out_of_range("object size beyond available buckets");
  }

  Region * allocate_new_region(size_t size) {
    auto bucket = get_log2_bin(size);
    auto obj_size = round_up_log2(size);
    asm("int3");
  }
  
  void attach_region(Region * region, size_t obj_size, int numa_node) {
    auto bucket = get_log2_bin(obj_size);
    if(bucket >= NUM_BUCKETS)
      throw std::out_of_range("object size beyond available buckets");
    _buckets[numa_node][bucket].push_back(region);
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
  inline unsigned get_log2_bin(size_t a) {
    unsigned fsmsb = unsigned(((sizeof(size_t) * 8) - __builtin_clzl(a)));
    if ((addr_t(1) << (fsmsb - 1)) == a) fsmsb--;
    return fsmsb;
  }

  inline size_t round_up_log2(size_t a) {
    int clzl = __builtin_clzl(a);
    int fsmsb = int(((sizeof(size_t) * 8) - clzl));
    if ((size_t(1) << (fsmsb - 1)) == a) fsmsb--;
    
    return (size_t(1) << fsmsb);
  }
}



private:
  nupm::Rca_AVL       _arena_allocator;
  std::list<Region *> _buckets[MAX_NUMA_ZONES][NUM_BUCKETS];
};

} // nupm
#endif
