/*
   Copyright [2017] [IBM Corporation]

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

#ifndef __CORE_AVL_MALLOC_H__
#define __CORE_AVL_MALLOC_H__

#if !defined(__cplusplus)
#error This is a C++ header
#endif

#include <common/types.h>
#include <common/chksum.h>
#include <common/stack.h>
#include <core/slab.h>
#include <vector>
#include "avl_tree.h"

namespace Core
{
/** 
 * Defines an allocated (free or not free) region of memory. This is 
 * basically the node of the AVL tree.  
 * 
 */
class Memory_region : public Core::AVL_node<Memory_region>
{
  friend class AVL_range_allocator;
  
 private:
  addr_t         _addr;
  size_t         _size;
  bool           _free = true;
  Memory_region* _next = nullptr;
  Memory_region* _prev = nullptr;
  char           _padding;
  //    size_t          _max_free = 0; // TODO: subtree pruning

 public:
  /** 
   * Constructor
   * 
   * @param addr Start of memory address
   * @param size Size of memory region in bytes
   * 
   */
  Memory_region(addr_t addr, size_t size) : _addr(addr), _size(size)
  {
  }

  /** 
   * Destructor
   * 
   * 
   * @return 
   */
  virtual ~Memory_region()
  {
  }

  /** 
   * Dump information about the region
   * 
   */
  void dump()
  {
    // if(_addr) {
    //   PLOG("node [%p]: addr=0x%lx size=%ld free=%s, chksum=%x", this, _addr, _size,
    //        _free ? "yes" : "no", Common::chksum32((void*)_addr, _size));
    // }
    // else {
    PLOG("node [%p]: addr=0x%lx size=%ld free=%s", this, _addr, _size, _free ? "yes" : "no");
  }

  /** 
   * Get start address of this region
   * 
   * 
   * @return Start address
   */
  addr_t addr()
  {
    return _addr;
  }

  /** 
   * Get size of allocation
   * 
   * 
   * @return 
   */
  size_t size() const
  {
    return _size;
  }


  /** 
   * Get whether region is free or not
   * 
   * 
   * @return True if region is free
   */
  bool is_free() const
  {
    return _free;
  }

 private:
  /** 
   * Comparator need by base class
   * 
   * @param n 
   * 
   * @return 
   */
  bool higher(AVL_node<Memory_region>* n)
  {
    return (_addr > ((Memory_region*) n)->_addr);
  }


  /** 
   * Helper to get left hand subtree
   * 
   * 
   * @return 
   */
  Memory_region* left()
  {
    return static_cast<Memory_region*>(subtree[Core::LEFT]);
  }

  /** 
   * Helper to get right hand subtree
   * 
   * 
   * @return 
   */
  Memory_region* right()
  {
    return static_cast<Memory_region*>(subtree[Core::RIGHT]);
  }


 protected:
  /** 
   * Find node with corresponding address
   * 
   * @param node 
   * @param addr 
   * 
   * @return 
   */
  Memory_region* find_region(Memory_region* node, addr_t addr)
  {
    if (node == nullptr) return nullptr;

#ifdef USE_RECURSION  // recursion breaks the stack for large trees

    if (node->_addr == addr) {
      return node;
    }
    else {
      Memory_region* result;
      if ((result = find_region(node->left(), addr))) {
        return result;
      }
      if ((result = find_region(node->right(), addr))) {
        return result;
      }
      return nullptr;
    }
#else
    std::vector<Memory_region*> stack;
    stack.push_back(node);

    while (!stack.empty()) {
      Memory_region* node = stack.back();
      stack.pop_back();

      if (node->_addr == addr) {
        return node;
      }
      else {
        Memory_region* l = node->left();
        if (l) stack.push_back(l);

        Memory_region* r = node->right();
        if (r) stack.push_back(r);
      }
    }

#endif
    return nullptr;
  }

  /** 
   * Find a free region that can contain a request block size. This
   * currently does a top-down, left-to-right, first-fit traversal. This
   * might not be the best traversal but it will do for now.
   * 
   * @param node 
   * @param size 
   * 
   * @return 
   */
  Memory_region* find_free_region(Memory_region* node, size_t size, size_t alignment = 0)
  {
    if (node == nullptr) return nullptr;

#ifdef USE_RECURSION

    if (node->_size >= size && node->_free) {
      if (alignment > 0) {
        if ((!(node->_addr & (alignment - 1UL)))) {
          return node;
        }
      }
      else {
        return node;
      }
    }

    Memory_region* result;
    if ((result = find_free_region(node->left(), size)) != nullptr)
      return result;
    else if ((result = find_free_region(node->right(), size)) != nullptr)
      return result;

#else
    //std::vector<Memory_region *> stack;
    Common::Fixed_stack<Memory_region*> stack;  // good for debugging
    stack.push(node);

    while (!stack.empty()) {
      Memory_region* node = stack.pop();

      if (node->_size >= size && node->_free) {
        if (alignment > 0) {
          if ((!(node->_addr & (alignment - 1UL)))) {
            return node;
          }
        }
        else {
          return node;
        }
      }
      else {
        Memory_region* l = node->left();
        if (l) stack.push(l);

        Memory_region* r = node->right();
        if (r) stack.push(r);
      }
    }
#endif

    return nullptr;
  }

  /** 
   * Find a free region that can contains requested address.
   * 
   * @param addr Address that should be contained
   * 
   * @return nullptr if not found, otherwise containing region
   */
  Memory_region* find_containing_region(Memory_region* node, addr_t addr)
  {
    if (node == nullptr) return nullptr;

#ifdef USE_RECURSION   
    if ((addr >= node->_addr) && (addr < (node->_addr + node->_size))) {  // to check
      return node;
    }
    else {
      Memory_region* result;
      if ((result = find_containing_region(node->left(), addr))) {
        return result;
      }
      if ((result = find_containing_region(node->right(), addr))) {
        return result;
      }
    }
#else
    Common::Fixed_stack<Memory_region*> stack;  // good for debugging
    stack.push(node);

    while (!stack.empty()) {
      Memory_region* node = stack.pop();

      if ((addr >= node->_addr) && (addr < (node->_addr + node->_size))) {  // to check
        return node;
      }
      else {
        Memory_region* l = node->left();
        if (l) stack.push(l);

        Memory_region* r = node->right();
        if (r) stack.push(r);
      }
    }    
#endif
    return nullptr; /* not found */
  }

} __attribute__((packed));


/** 
 * Simple AVL-based heap memory manager.  Template provides slab
 * allocator for node elements.
 * 
 */
class AVL_range_allocator
{
 private:
  static constexpr bool option_DEBUG = false;

  Core::Slab::CRuntime<Memory_region> __default_allocator;
  
  AVL_tree<Memory_region>* _tree;

 public:

    /** 
   * Constructor
   * 
   * @param slab Slab allocator for metadata
   * @param addr Start of memory address
   * @param size Size of memory slab in bytes
   * 
   */
  AVL_range_allocator(addr_t start, size_t size) : AVL_range_allocator(__default_allocator, start, size) {
  }

  /** 
   * Constructor
   * 
   * @param slab Slab allocator for metadata
   * @param addr Start of memory address
   * @param size Size of memory slab in bytes
   * 
   */
  AVL_range_allocator(Common::Base_slab_allocator& slab, addr_t start, size_t size) : _slab(slab)
  {
    assert(size > 0);

    /* establish root */
    Core::AVL_node<Core::Memory_region>** root    = nullptr;
    bool                                  newroot = true;

    if (slab.is_reconstructed()) {
      newroot = false;
      /* the first entry will be the root (atleast it should be!) */
      root = reinterpret_cast<Core::AVL_node<Core::Memory_region>**>(((addr_t) slab.get_first_element()));

      if (option_DEBUG) PLOG("reconstructed root pointer: %p", root);
    }
    else {
      /* create root pointer on slab */
      root = reinterpret_cast<Core::AVL_node<Core::Memory_region>**>(slab.alloc());

      if (option_DEBUG) PLOG("new root pointer: %p", (void*) root);

      *root = nullptr;
    }

    /* now we can create the tree and pass the root pointer */
    _tree = new AVL_tree<Memory_region>(root);

    if (newroot) {
      void* p = slab.alloc();

      if (!p) throw General_exception("AVL_range_allocator: failed to allocate from slab");

      if (option_DEBUG) PLOG("inserting root region (%lx-%lx)", start, start + size);

      try {
        _tree->insert_node(new (p) Memory_region(start, size));
      }
      catch (...) {
        PERR("inserting memory region (0x%lx-0x%lx) conflict detected.", start, start + size);
      }
    }
  }

  addr_t base()
  {
    /* update base */
    Memory_region* leftmost = leftmost_region();
    assert(leftmost);
    return leftmost->addr();
  }

  /** 
   * Destructor
   * 
   * 
   */
  virtual ~AVL_range_allocator()
  {
    /* delete node memory */
    _tree->apply_topdown([=](void* p) {        
        Memory_region* mr = static_cast<Memory_region*>(p);
        _slab.free(mr);
      });
    
    delete _tree;
  }

  /** 
   * Allocate a region of memory
   * 
   * @param size Size in bytes of region to allocate
   * @param alignment Alignment requirement in bytes
   * 
   * @return Pointer to 
   */
  Memory_region* alloc(size_t size, size_t alignment = 0)
  {
    // find fitting region
    Memory_region* root   = (Memory_region*) *(_tree->root());
    Memory_region* region = root->find_free_region(root, size, alignment);

    if (region == nullptr) {
      dump_info();
      throw General_exception("AVL_range_allocator: line %d failed to allocate %ld units", __LINE__, size);
    }

    assert(region->_size >= size);

    // exact fit
    if (region->_size == size) {
      region->_free = false;
    }
    else {
      void* p = _slab.alloc();
      if (!p) throw General_exception("AVL_range_allocator: line %d failed to allocate %ld units", __LINE__, size);


      Memory_region* left_over = new (p) Memory_region(region->_addr + size, region->_size - size);

      region->_size       = size;  // this fitting block will become the new allocation
      region->_free       = false;
      auto tmp            = region->_next;
      region->_next       = left_over;
      left_over->_prev    = region;
      left_over->_next    = tmp;
      if (tmp) tmp->_prev = left_over;

      _tree->insert_node(left_over);
    }

    if(region->_addr == 0x5dc0000) {
      PNOTICE("allocated at 5dc0000!!");
    }
    else {
      PLOG("allocated at:%lx", region->_addr);
    }
    return region;
  }

  /** 
   * Get the leftmost region in the tree (gives base of region)
   * 
   * 
   * @return Leftmost memory region
   */
  Memory_region* leftmost_region()
  {
    assert(_tree);
    Memory_region* r = (Memory_region*) *(_tree->root());
    if (r == nullptr) throw Logic_exception("AVL memory range tree root is nullptr");

    while (r->left()) r = r->left();
    return r;
  }

  /** 
   * Get the rightmost region in the tree
   * 
   * 
   * @return Rightmost memory region (give top end of region)
   */
  Memory_region* rightmost_region()
  {
    Memory_region* r     = (Memory_region*) *(_tree->root());
    while (r->right()) r = r->right();
    return r;
  }


  /** 
   * Allocate memory region at a specific location (used for rebuild). TODO needs thorough testing
   * 
   * @param addr Address to allocate at
   * @param size Region size
   * 
   * @return 
   */
  Memory_region* alloc_at(addr_t addr, size_t size)
  {
    // find fitting region
    Memory_region* root   = (Memory_region*) *(_tree->root());
    Memory_region* region = root->find_containing_region(root, addr);

    if (region == nullptr) {
      PWRN("alloc_at: cannot find containing region (addr=%lx, size=%ld)", addr, size);
      assert(0);
      return nullptr;
    }

    if (option_DEBUG) PLOG("alloc_at (addr=0x%lx,size=%ld) found fitting region %lx-%lx:", addr, size,
                           region->_addr, region->_addr + region->_size);

    assert(region->_size >= size);

    Memory_region* middle = nullptr;

    /* first split off front if needed and create middle chunk */
    if (region->_addr != addr) {
      assert(addr > region->_addr);
      auto left_size = addr - region->_addr;

      void* p = _slab.alloc();
      if (!p) throw General_exception("AVL_range_allocator: line %d failed to allocate %ld bytes", __LINE__, size);

      middle        = new (p) Memory_region(region->_addr + left_size, region->_size - left_size);
      middle->_size = region->_size - left_size;
      region->_size = left_size;  // make the containing region left chunk
      middle->_next = region->_next;
      middle->_prev = region;
      region->_next = middle;
      // region previous stays the same
      _tree->insert_node(middle);
    }
    else {
      middle = region;  // containing node becomes middle chunk (inserted)
    }

    middle->_free = false;  // middle chunk will become our allocated region

    /* now we have middle+right chunk combined; we must split off right-chunk */
    if (middle->_size > size) {
      auto  right_size = middle->_size - size;
      void* p          = _slab.alloc();
      if (!p) throw General_exception("AVL_range_allocator: line %d failed to allocate %ld bytes", __LINE__, size);

      Memory_region* right = new (p) Memory_region(middle->_addr + size, right_size);
      right->_size         = right_size;
      right->_next         = middle->_next;
      right->_prev         = middle;
      middle->_next        = right;
      middle->_size        = size;  // shrink middle chunk
      _tree->insert_node(right);
    }

    return middle;
  }

  /** 
   * Find a block with corresponding start address
   * 
   * @param addr Start address
   * 
   * @return Pointer to memory region if found, otherwise nullptr
   */
  Memory_region* find(addr_t addr)
  {
    Memory_region* root = (Memory_region*) *(_tree->root());
    return root->find_region(root, addr);
  }


  /** 
   * Add a new memory region to the allocator
   * 
   * @param addr Base address of region
   * @param size Size of region in bytes
   */
  void add_new_region(addr_t addr, size_t size)
  {
    assert(addr > 0);
    assert(size > 0);
    void* p = _slab.alloc();
    if (!p) throw General_exception("AVL_range_allocator: line %d failed to allocate %ld bytes", __LINE__, size);

    auto new_region = new (p) Memory_region(addr, size);
    assert(new_region);
    _tree->insert_node(new_region);

    //    update_base_from_tree();
  }


  /** 
   * Free an allocated region
   * 
   * @param addr Start address of the memory region
   * 
   * @return Number of bytes freed
   */
  size_t free(addr_t addr)
  {
    Memory_region* region = find(addr);
    if (region == nullptr) {
      PERR("invalid call to free: bad address");
      return -1;
    }

    size_t return_count = region->_size;
    region->_free       = true;

    // right coalesce
    {
      Memory_region* right = region->_next;
      if (right && right->_free) {
        region->_size += right->_size;
        region->_next                         = right->_next;
        if (right->_next) right->_next->_prev = region;
        _tree->remove_node(right);

        // release node memory
        _slab.free(right);
      }
    }

    // left coalesce
    {
      Memory_region* left = region->_prev;
      if (left && left->_free) {
        left->_size += region->_size;
        left->_next                             = region->_next;
        if (region->_next) region->_next->_prev = left;
        _tree->remove_node(region);

        // release node memory
        _slab.free(region);
      }
    }

    return return_count;
  }

  /** 
   * Dump the tree for debugging purposes
   * 
   */
  void dump_info()
  {
    assert(_tree->root());
    AVL_tree<Memory_region>::dump(*(_tree->root()));
  }

  /** 
   * Apply function to each region
   * 
   * @param functor 
   */
  void apply(std::function<void(addr_t, size_t, bool)> functor)
  {
    _tree->apply_topdown([functor](void* p) {
      Memory_region* mr = static_cast<Memory_region*>(p);
      functor(mr->addr(), mr->size(), mr->is_free());
    });
  }

  // inline addr_t base()
  // {
  //   if(_base == 0)
  //     update_base_from_tree();

  //   return _base;
  // }

 private:
  /* NOTE: specifically no members that will be on the stack */

  Common::Base_slab_allocator& _slab; /**< volatile slab allocator for metadata */
  //  addr_t _base;
};

/** 
 * AVL-based heap memory manager with transparency. This class ties everything together.
 * This class requires that the slab allocator and region are explicitly passed to the 
 * constructor.  This should allow the use of memory allocated for SPDK IO buffers etc.
 * 
 */
class Arena_allocator : public Common::Base_memory_allocator
{
 public:
  /** 
   * Constructor
   * 
   * @param metadata_slab Slab allocator for AVL tree nodes
   * @param region Region of memory to manage
   * @param region_size Size of memory region
   * 
   */
  Arena_allocator(Common::Base_slab_allocator& metadata_slab, void* region, size_t region_size)
    : _range_allocator(metadata_slab, (addr_t) region, region_size)
  {
    if (!region) throw Constructor_exception("nullptr 'region' parameter in Arena_allocator");

    if (region_size == 0) throw Constructor_exception("Zero 'region_size' parameter in Arena_allocator");
  }


  /** 
   * Allocate a variable sized block of memory
   * 
   * @param size Size to allocate in bytes
   * @param numa_node 
   * @param alignment 
   * 
   * @return Pointer to newly allocated memory.
   */
  void* alloc(size_t size, int numa_node = -1, size_t alignment = 0)
  {
    if (size == 0) size = 1;

    Memory_region* mr;

    try {
      mr = _range_allocator.alloc(size);
    }
    catch (General_exception e) {
      PERR("Arena_allocator: out of memory");
      exit(0);
    }
    assert(mr);
    return (void*) mr->addr();
  }


  /** 
   * Free a previously allocated block
   * 
   * @param ptr Pointer to block
   */
  size_t free(void* ptr)
  {
    assert(ptr);
    return _range_allocator.free((addr_t) ptr);
  }

  /** 
   * Get the size of an allocation
   * 
   * @param ptr Pointer to allocated block
   * 
   * @return Size of allocation in bytes
   */
  size_t get_size(void* ptr)
  {
    assert(ptr);
    const Core::Memory_region* region = _range_allocator.find((addr_t) ptr);
    if (region == nullptr)
      return -1;
    else
      return region->size();
  }

  /** 
   * Get free space capacity in bytes
   * 
   * 
   * @return Unused space in bytes
   */
  size_t free_space()
  {
    size_t free_bytes = 0;
    apply([&free_bytes](addr_t a, size_t s, bool is_free) {
      if (is_free) free_bytes += s;
    });
    return free_bytes;
  }


  /** 
   * Used to determine how many entries the slab allocator should support
   * 
   * @param entries 
   * 
   * @return 
   */
  static size_t recommend_metadata_slab_size(size_t entries)
  {
    // TODO
    return 0;
  }

  /** 
   * Output debugging information
   * 
   */
  void dump_info()
  {
    _range_allocator.dump_info();
  }

  /** 
   * Apply functor to elements of the tree
   * 
   * @param functor 
   */
  void apply(std::function<void(addr_t, size_t, bool)> functor)
  {
    _range_allocator.apply(functor);
  }

  /** 
   * Add a new region to the allocator
   * 
   * @param vaddr Base address (virtual)
   * @param size Size of region in bytes
   */
  void add_new_region(void* vaddr, size_t size)
  {
    _range_allocator.add_new_region((addr_t) vaddr, size);
  }

  /** 
   * Gives the upper limit on memory used by the heap. This is
   * done by looking up the rightmost node in the tree.
   * 
   * 
   * @return Upper limit in bytes
   */
  size_t used_zone_limit()
  {
    Memory_region* r = _range_allocator.rightmost_region();

    if (r->is_free())
      return r->addr() - _range_allocator.base();
    else
      return r->addr() + r->size() - _range_allocator.base();
  }


 private:
  Core::AVL_range_allocator _range_allocator;
};



/** 
 * AVL-based heap region manager.  This uses the C-runtime heap for slab purposes
 * and therefore is only used as an in-memory structure.  
 * 
 */
class Region_allocator
{
 public:
  /** 
   * Constructor
   * 
   * @param region Region of memory to manage
   * @param region_size Size of memory region
   * 
   */
  Region_allocator(addr_t region, size_t region_size) : _slab(), _range_allocator(_slab, region, region_size)
  {
    if (region_size == 0) throw Constructor_exception("invalid 'region' parameter");
  }


  /** 
   * Allocate a variable sized block of memory
   * 
   * @param size Size to allocate in bytes
   * @param alignment Alignment in bytes
   * 
   * @return Pointer to newly allocated memory.
   */
  addr_t alloc(size_t size, size_t alignment)
  {
    if (size == 0) size = 1;

    Memory_region* mr = _range_allocator.alloc(size);

    if (mr == nullptr) throw General_exception("no memory.");

    return mr->addr();
  }

  /** 
   * Allocate a region at a specific location
   * 
   * @param region 
   * @param region_size 
   * 
   * @return 
   */
  status_t alloc_at(addr_t region, size_t region_size)
  {
    auto r = _range_allocator.alloc_at(region, region_size);
    return r == nullptr ? E_FAIL : S_OK;
  }

  /** 
   * Get memory region for a specific address
   * 
   * @param region 
   * 
   * @return Pointer to memory region node
   */
  const Core::Memory_region* contains(addr_t region)
  {
    return (const Core::Memory_region*) _range_allocator.find(region);
  }


  /** 
   * Free a previously allocated block
   * 
   * @param ptr Pointer to block
   */
  void free(addr_t addr)
  {
    _range_allocator.free(addr);
  }

  /** 
   * Debugging dump
   * 
   */
  void dump_info()
  {
    _range_allocator.dump_info();
  }

 private:
  Core::Slab::CRuntime<Core::Memory_region> _slab;
  Core::AVL_range_allocator                 _range_allocator;
};
}

#endif  //__CORE_AVL_MALLOC_H__
