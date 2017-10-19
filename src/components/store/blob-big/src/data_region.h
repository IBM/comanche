#ifndef __DATA_REGION_H__
#define __DATA_REGION_H__

#include <core/slab.h>
#include <vector>
#include <core/avl_tree.h>

namespace Blob
{

class Data_region : public Core::AVL_node<Data_region>
{
  template <class T> friend class Data_region_tree;

private:
  addr_t         _addr;
  size_t         _size;
  bool           _free = true;
  bool           _head = false;
  Data_region*   _next = nullptr;
  Data_region*   _prev = nullptr;

  //    size_t          _max_free = 0; // TODO: subtree pruning

public:
  /** 
   * Constructor
   * 
   * @param addr Start of memory address
   * @param size Size of memory region in bytes
   * 
   */
  Data_region(addr_t addr, size_t size) : _addr(addr), _size(size)
  {
  }

  /** 
   * Destructor
   * 
   * 
   * @return 
   */
  virtual ~Data_region()
  {
  }

  /** 
   * Dump information about the region
   * 
   */
  void dump()
  {
    PLOG("node [%p]: addr=0x%lx size=%ld free=%s",this, _addr, _size, _free ? "yes" : "no");
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

  /** 
   * Get whether region is head or not
   * 
   * 
   * @return 
   */
  bool is_head() const
  {
    return _head;
  }

private:
  /** 
   * Comparator need by base class
   * 
   * @param n 
   * 
   * @return 
   */
  bool higher(Core::AVL_node<Data_region>* n)
  {
    return (_addr > ((Data_region*)n)->_addr);
  }


  /** 
   * Helper to get left hand subtree
   * 
   * 
   * @return 
   */
  Data_region* left()
  {
    return static_cast<Data_region*>(subtree[Core::LEFT]);
  }

  /** 
   * Helper to get right hand subtree
   * 
   * 
   * @return 
   */
  Data_region* right()
  {
    return static_cast<Data_region*>(subtree[Core::RIGHT]);
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
  Data_region* find_region(Data_region* node, addr_t addr)
  {
    if (node == nullptr) return nullptr;

#ifdef USE_RECURSION // recursion breaks the stack for large trees
    
    if (node->_addr == addr) {
      return node;
    }
    else {
      Data_region* result;
      if ((result = find_region(node->left(), addr))) {
        return result;
      }
      if ((result = find_region(node->right(), addr))) {
        return result;
      }
      return nullptr;
    }
#else
    std::vector<Data_region *> stack;
    stack.push_back(node);

    while(!stack.empty()) {

      Data_region * node = stack.back();
      stack.pop_back();

      if (node->_addr == addr) {
        return node;        
      }
      else {
        Data_region * l = node->left();
        if(l) stack.push_back(l);

        Data_region * r = node->right();
        if(r) stack.push_back(r);
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
  Data_region* find_free_region(Data_region* node, size_t size, size_t alignment = 0)
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

    Data_region* result;
    if ((result = find_free_region(node->left(), size)) != nullptr)
      return result;
    else if ((result = find_free_region(node->right(), size)) != nullptr)
      return result;

#else
    //std::vector<Data_region *> stack;
    Common::Fixed_stack<Data_region *> stack; // good for debugging
    stack.push(node);

    while(!stack.empty()) {

      Data_region * node = stack.pop();
      
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
        Data_region * l = node->left();
        if(l) stack.push(l);

        Data_region * r = node->right();
        if(r) stack.push(r);
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
  Data_region* find_containing_region(Data_region* node, addr_t addr)
  {
    if (node == nullptr) return nullptr;

    if ((addr >= node->_addr) && (addr < (node->_addr + node->_size))) {  // to check
      return node;
    }
    else {
      Data_region* result;
      if ((result = find_containing_region(node->left(), addr))) {
        return result;
      }
      if ((result = find_containing_region(node->right(), addr))) {
        return result;
      }
    }
    return nullptr;
  }

public:
  /** 
   * Mark as a 'head' region
   * 
   */
  void set_as_head()
  {
    _head = true;
  }


} __attribute__((packed));



/** 
 * Simple AVL-based heap memory manager.  Template provides node type
 * 
 */
template <class Node = Data_region>
class Data_region_tree
{
private:
  static constexpr bool option_DEBUG = true;

public:
  
  /** 
   * Constructor
   * 
   * @param slab Slab allocator for metadata
   * @param addr Start of memory address
   * @param size Size of memory slab in bytes
   * 
   */
  Data_region_tree(Common::Base_slab_allocator& slab,
                      addr_t start,
                      size_t size) : _slab(slab)
  {
    assert(size > 0);

    /* establish root */
    Core::AVL_node<Node>** root    = nullptr;
    bool                                  newroot = true;

    if (slab.is_reconstructed()) {
      newroot = false;
      /* the first entry will be the root (atleast it should be!) */
      root =
        reinterpret_cast<Core::AVL_node<Node>**>(((addr_t)slab.get_first_element()));

      if (option_DEBUG) PLOG("reconstructed root pointer: %p", root);
    }
    else {
      /* create root pointer on slab */
      root = reinterpret_cast<Core::AVL_node<Node>**>(slab.alloc());

      if (option_DEBUG) PLOG("new root pointer: %p", (void*)root);
      *root = nullptr;
    }

    /* now we can create the tree and pass the root pointer */
    _tree = new Core::AVL_tree<Node>(root);

    if (newroot) {
      void* p = slab.alloc();

      if(!p)
        throw General_exception("Data_region_tree: failed to allocate from slab");

      if (option_DEBUG)
        PLOG("inserting root region (%lx-%lx)", start, start + size);

      try {
        _tree->insert_node(new (p) Node(start, size));
      }
      catch(...) {
        PERR("inserting memory region (0x%lx-0x%lx) conflict detected.",start, start+size);
      }
             
    }
  }

  addr_t base()
  {
    /* update base */
    Node * leftmost = leftmost_region();
    assert(leftmost);
    return leftmost->addr();
  }

  /** 
   * Destructor
   * 
   * 
   */
  virtual ~Data_region_tree()
  {
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
  Node* alloc(size_t size, size_t alignment = 0)
  {
    // find fitting region
    Node* root   = (Node*)*(_tree->root());
    Node* region = root->find_free_region(root, size, alignment);

    if (region == nullptr) {
      PERR("unable to find free region (size=%ld)", size);
      throw General_exception("Data_region_tree: line %d failed to allocate %ld bytes", __LINE__, size);
    }

    assert(region->_size >= size);

    // exact fit
    if (region->_size == size) {
      region->_free = false;
    }
    else {
      void * p = _slab.alloc();
      if(!p)
        throw General_exception("Data_region_tree: line %d failed to allocate %ld bytes", __LINE__, size);


      Node* left_over =
        new (p) Node(region->_addr + size, region->_size - size);

      region->_size       = size;  // this fitting block will become the new allocation
      region->_free       = false;
      auto tmp            = region->_next;
      region->_next       = left_over;
      left_over->_prev    = region;
      left_over->_next    = tmp;
      if (tmp) tmp->_prev = left_over;
      
      _tree->insert_node(left_over);
    }

    return region;
  }

  /** 
   * Get the leftmost region in the tree (gives base of region)
   * 
   * 
   * @return Leftmost memory region
   */
  Node* leftmost_region()
  {
    assert(_tree);
    Node* r = (Node*)*(_tree->root());
    if(r == nullptr) throw Logic_exception("AVL memory range tree root is nullptr");
    
    while(r->left())
      r = r->left();
    return r;
  }

  /** 
   * Get the rightmost region in the tree
   * 
   * 
   * @return Rightmost memory region (give top end of region)
   */
  Node* rightmost_region()
  {
    Node* r = (Node*)*(_tree->root());
    while(r->right())
      r = r->right();
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
  Node* alloc_at(addr_t addr, size_t size)
  {
    // find fitting region
    Node* root   = (Node*)*(_tree->root());
    Node* region = root->find_containing_region(root, addr);

    if (region == nullptr) {
      assert(0);
      return nullptr;
    }

    if (option_DEBUG)
      PLOG("alloc_at (addr=0x%lx,size=%ld) found fitting region %p:", addr, size, region);

    assert(region->_size >= size);
    if (addr > 0x200400UL) {
      assert(region->_free == true);
    }

    Node* middle = nullptr;

    /* first split off front if needed and create middle chunk */
    if (region->_addr != addr) {
      assert(addr > region->_addr);
      auto left_size = addr - region->_addr;

      void * p = _slab.alloc();
      if(!p)
        throw General_exception("Data_region_tree: line %d failed to allocate %ld bytes", __LINE__, size);

      middle = new (p) Node(region->_addr + left_size, region->_size - left_size);
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
      auto           right_size = middle->_size - size;
      void * p = _slab.alloc();
      if(!p)
        throw General_exception("Data_region_tree: line %d failed to allocate %ld bytes", __LINE__, size);

      Node* right = new (p) Node(middle->_addr + size, right_size);
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
  Node* find(addr_t addr)
  {
    Node* root = (Node*)*(_tree->root());
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
    void * p = _slab.alloc();
    if(!p)
      throw General_exception("Data_region_tree: line %d failed to allocate %ld bytes", __LINE__, size);
      
    auto new_region = new (p) Node(addr, size);
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
    Node* region = find(addr);
    if (region == nullptr) {
      PERR("invalid call to free: bad address");
      return -1;
    }

    size_t return_count = region->_size;
    region->_free       = true;

    // right coalesce
    {
      Node* right = region->_next;
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
      Node* left = region->_prev;
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
   * Trim an allocated region
   * 
   * @param addr Start address of the memory region
   * 
   * @return Number of bytes trimmed
   */
  size_t trim(addr_t addr, size_t trim_size)
  {
    Node* region = find(addr);
    if (region == nullptr) {
      PERR("invalid call to free: bad address");
      return -1;
    }

    if(trim_size > region->_size)
      throw API_exception("trim failed; needs expand");
    
    size_t trimmed = region->_size - trim_size;

    // give trim remainder to right node if free
    {
      Node* right = region->_next;
      if (right && right->_free) {
        right->_size += trimmed;
        right->_addr -= trimmed;
        return trimmed;
      }
    }

    // otherwise create a new free node
    void * p = _slab.alloc();
    if(!p)
      throw General_exception("Data_region_tree: %s:%d failed to allocate node", __FILE__, __LINE__);

    Node* newright = new (p) Node(region->_addr + trim_size, trimmed);
    newright->_next = region->_next;
    newright->_prev = region;
    newright->_free = true;
    region->_next = newright;
    _tree->insert_node(newright);

    return trimmed;
  }


  /** 
   * Dump the tree for debugging purposes
   * 
   */
  void dump_info()
  {
    assert(_tree->root());
    Core::AVL_tree<Node>::dump(*(_tree->root()));
  }

  /** 
   * Apply function to each region
   * 
   * @param functor 
   */
  void apply(std::function<void(addr_t, size_t, bool,bool)> functor)
  {
    _tree->apply_topdown([functor](void* p) {
        Node* mr = static_cast<Node*>(p);
        functor(mr->addr(), mr->size(), mr->is_free(),mr->is_head());
      });    
  }

private:
  /* NOTE: specifically no members that will be on the stack */
  Core::AVL_tree<Node>* _tree;

  Common::Base_slab_allocator& _slab; /**< volatile slab allocator for metadata */

};

}

#endif
