#ifndef __NUPM_ARENA_ALLOC_H__
#define __NUPM_ARENA_ALLOC_H__

#include <set>
#include <mutex>
#include "rpmalloc.h"
#include "nd_utils.h"
#include "block_bitmap.h"

namespace nupm
{

/** 
 * Coarse-grained allocator.  Checks for invalid frees. Lock per numa socket.
 * 
 */
class Arena_allocator_volatile : private ND_control
{
private:
  static constexpr unsigned MAX_NUMA_SOCKETS = 4;
  
public:
  Arena_allocator_volatile(unsigned granularity = GB(1)) : _granularity(granularity) {

    if(granularity % 4096 > 0)
      throw Constructor_exception("invalid granularity");
    
    std::vector<std::pair<void *, size_t>> regions(_n_sockets);

    /* determine contiguous regions */
    for(unsigned s=0; s<_n_sockets; s++) {

      unsigned long region_start = 0;
      unsigned long region_end = 0;
      
      for(auto& m: _mappings[s]) {
        if(region_start == 0) {
          region_start = ((unsigned long)m.first);
          region_end = region_start + m.second;
          _vmr_bases[s] = region_start;
        }
        else if(region_end == ((unsigned long)m.first)) {
          region_end += m.second;
        }
        else throw Logic_exception("unexpected condition");

        regions[s] = std::make_pair((void*)region_start,
                                    region_end - region_start);
      }

      _vmr_ends[s] = _vmr_bases[s] + regions[s].second;

      /* create free page list */
      size_t num_pages = (regions[s].second / _granularity) + 1;
      byte * page = (byte *) regions[s].first;
      PLOG("num_pages:%lu numa-zone:%u", num_pages, s);
      for(size_t p = 0 ; p < num_pages ; p++) {
        _free_pages[s].insert(&page[p * _granularity]);
      }
    }   
  }

  /** 
   * Allocate a page
   * 
   * @param numa_node 
   * 
   * @return Pointer to page or throw exception on empty.
   */
  void * alloc(int numa_node) {
    if(numa_node < 0)
      numa_node = 0;
    
    if(((unsigned)numa_node) > _n_sockets)
      throw API_exception("numa_node out of bounds");

    void * result = nullptr;

    {
      std::lock_guard<std::mutex> g(_free_pages_lock[numa_node]);
      if(_free_pages[numa_node].empty())
        throw General_exception("arena allocator out of pages.");    
      
      auto iter = _free_pages[numa_node].begin();
      result = *iter;
      _free_pages[numa_node].erase(iter);
    }
    std::lock_guard<std::mutex> g(_used_pages_lock[numa_node]);
    _used_pages[numa_node].insert(result);
    
    return result;
  }

  /** 
   * Free a previously allocated page
   * 
   * @param ptr Pointer to region to free.
   */
  void free(void * ptr) {
    unsigned long p = ((unsigned long) ptr);
    unsigned numa_node;
    bool found = false;
    for(numa_node=0; numa_node<_n_sockets; numa_node++) {
      if(p >= _vmr_bases[numa_node] && p < _vmr_ends[numa_node]) {
        found = true;
        break;
      }
    }
    if(!found)
      throw API_exception("invalid pointer passed to free");

    {
      std::lock_guard<std::mutex> g(_used_pages_lock[numa_node]);
      auto up = _used_pages[numa_node].find(ptr);
      if(up == _used_pages[numa_node].end())
        throw API_exception("invalid pointer passed to free; not in used map");
      _used_pages[numa_node].erase(up);
    }

    std::lock_guard<std::mutex> g(_free_pages_lock[numa_node]);
    _free_pages[numa_node].insert(ptr);
  }

  void free(unsigned numa_node, void * ptr) {
    {
      std::lock_guard<std::mutex> g(_used_pages_lock[numa_node]);
      auto up = _used_pages[numa_node].find(ptr);
      if(up == _used_pages[numa_node].end())
        throw API_exception("invalid pointer passed to free; not in used map");
      _used_pages[numa_node].erase(up);
    }

    std::lock_guard<std::mutex> g(_free_pages_lock[numa_node]);
    _free_pages[numa_node].insert(ptr);
  }

  
private:
  size_t                 _granularity;
  Common::Std_allocator  _std_allocator;
  unsigned long          _vmr_bases[MAX_NUMA_SOCKETS];
  unsigned long          _vmr_ends[MAX_NUMA_SOCKETS];

  std::set<void *> _free_pages[MAX_NUMA_SOCKETS];
  std::set<void *> _used_pages[MAX_NUMA_SOCKETS];
  std::mutex       _free_pages_lock[MAX_NUMA_SOCKETS];
  std::mutex       _used_pages_lock[MAX_NUMA_SOCKETS];
};

} // namespace nupm

#endif // __NUPM_ARENA_ALLOC_H__
