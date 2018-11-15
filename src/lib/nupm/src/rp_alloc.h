#ifndef __NUPM_RP_ALLOC_H__
#define __NUPM_RP_ALLOC_H__

#include "rpmalloc.h"
#include "nd_utils.h"
#include "arena_alloc.h"



namespace nupm
{

template <unsigned NUMA_NODE>
class Rp_allocator_volatile
{
  static constexpr size_t ARENA_PAGE_SIZE = GiB(1); //MiB(64);
public:
  /** 
   * Constructor
   * 
   */
  Rp_allocator_volatile() {

    _rpmalloc_config = {
      hook_memory_map,
      hook_memory_unmap,
      ARENA_PAGE_SIZE, // page size
      1,  // span size
      1024,
      0};
    
    rpmalloc_initialize_config(&_rpmalloc_config);
      
    initialize_thread();
  }

  void * alloc(size_t size, int numa_node = 0) {
    return rpmalloc(size);
  }

  void free(void * ptr) {
    rpfree(ptr);
  }

  /** 
   * Call on each thread before using this API
   * 
   */
  inline void initialize_thread() {
    rpmalloc_thread_initialize();
  }

  /** 
   * Call to clean up thread state
   * 
   */
  inline void finalize_thread() {
    rpmalloc_finalize();
  }
  
private:
  static void* hook_memory_map(size_t size, size_t* offset) {
    void * ptr = _arena.alloc(NUMA_NODE);
    PLOG("hook_memory_map: %p size=%lu MiB offset*=%p numa=%d",
         ptr,
         REDUCE_MiB(size),
         offset,
         NUMA_NODE);
    return ptr;
  }

  static void hook_memory_unmap(void* address, size_t size, size_t offset, size_t release) {
    PLOG("hook_memory_unmap: size=%lu offset=%lu release=%lu", size, offset, release);
    _arena.free(address);
  }

  
private:
  static Arena_allocator_volatile _arena; /* usually 1GiB region allocation */
  rpmalloc_config_t               _rpmalloc_config;
};

} // namespace nupm

#endif // __NUPM_PAGE_ALLOC_H__
