/*
   eXokernel Development Kit (XDK)

   Samsung Research America Copyright (C) 2013
   IBM Research Copyright (C) 2019

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.
*/
#ifndef __COMMON_MEMORY_H__
#define __COMMON_MEMORY_H__

#include <assert.h>
#include <common/logging.h>
#include <common/types.h>

#if defined(__linux__)
#include <malloc.h>
#elif defined(__MACH__)
#include <malloc/malloc.h>
#endif

#include <stdlib.h>
#include <sys/mman.h>

namespace Common
{
/**
 * Allocate memory at a specific region.  Mainly for debugging purposes.
 *
 * @param size Size of region to allocate in bytes.
 * @param addr Location to allocate at.
 *
 * @return
 */
void *malloc_at(size_t size, addr_t addr = 0);

void free_at(void *ptr);

/**
 * Generic slab memory interface
 *
 *
 * @return
 */
class Base_slab_allocator {
 public:
  /**
   * Allocate slab element
   *
   *
   * @return Pointer to new element
   */
  virtual void *alloc() = 0;

  /**
   * Free slab element
   *
   * @return 0 on success
   */
  virtual size_t free(void *) = 0;

  /**
   * Determine if slab was reconstructed
   *
   *
   * @return True if reconstructed
   */
  virtual bool is_reconstructed() = 0;

  /**
   * Get a pointer to the first element of the slab. Needed for rebuilds.
   *
   *
   * @return Pointer to first element.
   */
  virtual void *get_first_element() {
    PWRN("get_first_element not implemented");
    return nullptr;
  }

  virtual void dump_info() { PWRN("dump_info not implemented"); }
};

/**
 * Generic memory allocation interface
 *
 */
class Base_memory_allocator {
 public:
  /**
   * Allocate memory
   *
   * @param size Size to allocate in bytes
   * @param numa_node NUMA node identifier
   * @param alignment Alignment of allocation in bytes
   *
   * @return Pointer to allocated memory
   */
  virtual void *alloc(size_t size,
                      int numa_node = -1,
                      size_t alignment = 0) = 0;

  /**
   * Free previously allocated memory
   *
   * @param ptr Pointer to allocated region
   */
  virtual size_t free(void *ptr) = 0;
};


/**
 * Memory allocator that can be reconstituted
 *
 */
class Reconstituting_allocator {
 public:

  /** 
   * Allocate region of memory
   * 
   * @param size 
   * @param numa_node 
   * @param alignment 
   * 
   * @return 
   */
  virtual void *alloc(size_t size,
                      int numa_node = -1,
                      size_t alignment = 0) = 0;

  /** 
   * Free region of memory
   * 
   * @param p Pointer to memory to free
   * @param numa_node NUMA node memory is associated with
   * @param size Size of region to free (providing this improves performance)
   */
  virtual void free(void *ptr, int numa_node, size_t size = 0) = 0;
  
  /** 
   * Inject previous allocation (for rebuild)
   * 
   * @param size Size of allocation in bytes
   * @param numa_node NUMA node
   * @param alignment Alignment
   */
  virtual void inject_allocation(void * ptr, size_t size, int numa_node) = 0;
};
  

/**
 * Generic heap allocator interface
 *
 *
 */
class Base_heap_allocator : public Base_memory_allocator {
 public:
  enum {
    HEAP_TYPE_UNSPECIFIED = 0,
    HEAP_TYPE_NVME_FIXED = 1,
    HEAP_TYPE_NVME_DUNE_PAGING = 2,
  };

  /**
   * Determine if slab was reconstructed
   *
   *
   * @return True if reconstructed
   */
  virtual bool is_reconstructed() = 0;

  /**
   * Get first element/base of heap
   *
   *
   * @return Pointer to base of heap
   */
  virtual void *base() = 0;

  /**
   * Get type of heap
   *
   *
   * @return Enumeration type (see above)
   */
  virtual int type() const { return HEAP_TYPE_UNSPECIFIED; }

  /**
   * Flush data on the slab
   *
   */
  virtual void flush() = 0;

  /**
   * Dump debugging information
   *
   */
  virtual void dump_info(){};

  /**
   * Return free space in bytes
   *
   */
  virtual size_t free_space() { return 0; };
};

/**
 * Standard allocator based on POSIX calls
 *
 * @param size
 * @param numa_node
 * @param alignment
 *
 * @return
 */
class Std_allocator : public Base_memory_allocator {
 public:
  Std_allocator() {}

  void *alloc(size_t size, int numa_node = -1, size_t alignment = 0) {
    assert(numa_node == -1);
    if (alignment > 0) {
      void *ptr;
      int rc = posix_memalign(&ptr, alignment, size);
      if (rc) return nullptr;
      return ptr;
    }
    else {
      return malloc(size);
    }
  }

  size_t free(void *ptr) {
    size_t size;
    assert(ptr);
#if defined(__linux__)
    size = malloc_usable_size(ptr);
#elif defined(__MACH__)
    size = malloc_size(ptr);
#else
    size = 0;
#endif

    ::free(ptr);
    return size;
  }

  void *realloc(void *ptr, size_t size) { return ::realloc(ptr, size); }
};
}  // namespace Common
#endif
