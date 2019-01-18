#ifndef __CORE_HEAP_ALLOCATOR_H__
#define __CORE_HEAP_ALLOCATOR_H__

#if !defined(__cplusplus)
#error This is a C++ header
#endif

#include <core/slab.h>
#include <core/avl_malloc.h>
#include <memory>

namespace Core
{

template <class T=void>
class Heap_allocator : public std::allocator<T>
{
  static constexpr size_t MAX_SLAB_SLOTS = 2048;
  
 public:
  Heap_allocator(void * region,
                 size_t region_size,
                 const std::string label)
      :
      _slab_size(Core::Slab::Allocator<>::determine_size(MAX_SLAB_SLOTS)),
      _slab(region, _slab_size, label, false),
      _arena_start(reinterpret_cast<void *>(((addr_t)region) + _slab_size)),
      _arena_size(region_size = _slab_size),
      _arena(_slab, _arena_start, _arena_size)
  {
    PLOG("Heap_allocator: slab size=%lu KiB", REDUCE_KiB(_slab_size));
    PLOG("Heap_allocator: arena @ %p", _arena_start);
  }

  T * allocate(size_t s) {
    return (T *) _arena.alloc(s);
  }

 private:
  const size_t                         _slab_size;
  Core::Slab::Allocator<Memory_region> _slab; /*< slab allocator */
  const void *                         _arena_start; /*< managed arena */
  const size_t                         _arena_size;
  Core::Arena_allocator                _arena; /*< arena manager */
};

}

#endif // __CORE_HEAP_ALLOCATOR_H__
