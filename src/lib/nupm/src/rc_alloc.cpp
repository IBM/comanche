#include <numa.h>
#include <stdexcept>
#include <common/logging.h>
#include <common/exceptions.h>
#include <core/slab.h>
#include <core/avl_malloc.h>

#include "rc_alloc.h"
#include "memkind_allocator.h"


/*< enabling use memkind allocator without reconstitution */
#define RCA_USE_MEMKIND 

namespace nupm
{

namespace Rca
{
  int max_numa_node;
}  

__attribute__((constructor))
static void init_Rca() 
{
  Rca::max_numa_node = numa_max_node();
};


class Rca_AVL_internal
{
 public:
  Rca_AVL_internal()
#ifndef RCA_USE_MEMKIND 
      : _slab()
#endif
  {
    _allocators.reserve(Rca::max_numa_node + 1);
    for(int i=0;i<=Rca::max_numa_node; i++)
      _allocators[i] = 0;

#ifdef RCA_USE_MEMKIND
    
#endif
  }
  
  ~Rca_AVL_internal() {
    for(auto& i : _allocators) {
      if(i != nullptr) delete i;
    }
  }

  void add_managed_region(const std::string& pmem_file, int numa_node)
  {
#ifndef RCA_USE_MEMKIND
    throw API_exception("add_managed_region pmem_file not supported");
#endif
    PLOG("adding memkind (%s,%d)", pmem_file.c_str(), numa_node);
    
    if(_allocators[numa_node])
      throw API_exception("add_managed_region can only be called once per numa node with memkind");

    _allocators[numa_node] = new Memkind_allocator(pmem_file);
  }
  
  void add_managed_region(void * region_base, size_t region_length, int numa_node)
  {
#ifdef RCA_USE_MEMKIND
    throw API_exception("add_managed_region with base,length, node not supported for memkind");
#else
    if(_allocators[numa_node] == nullptr) {
      
      _allocators[numa_node] = new Core::AVL_range_allocator(_slab,
                                                             reinterpret_cast<addr_t>(region_base),
                                                             region_length);
    }
    else {
      _allocators[numa_node]->add_new_region(reinterpret_cast<addr_t>(region_base), region_length);
    }
#endif
  }

  void inject_allocation(void * ptr, size_t size, int numa_node)
  {
    assert(ptr);
    auto mrp = _allocators[numa_node]->alloc_at(reinterpret_cast<addr_t>(ptr), size);
    if(mrp == nullptr)
      throw General_exception("alloc_at on AVL range allocator failed unexpectedly");    
  }
  
  void * alloc(size_t size,
               int numa_node,
               size_t alignment)
  {
    void * p = nullptr;
    if(_allocators[numa_node]->posix_memalign(&p, size, alignment))
      throw General_exception("alloc failed");
    return p;    
  }

  void free(void * ptr,
            int numa_node)
  {
    _allocators[numa_node]->free(ptr);
  }
  

  
 private:

#ifndef RCA_USE_MEMKIND 
  Core::Slab::CRuntime<Core::Memory_region> _slab; /* use C runtime for slab? */
  std::vector<Core::AVL_range_allocator *>  _allocators;
#else
  std::vector<nupm::Memkind_allocator *>  _allocators;
#endif

  
};

Rca_AVL::Rca_AVL() : _rca(new Rca_AVL_internal())
{
}

Rca_AVL::~Rca_AVL() {
  delete _rca;
}

void Rca_AVL::add_managed_region(void * region_base, size_t region_length, int numa_node)
{
  if(numa_node > Rca::max_numa_node)
    throw std::invalid_argument("numa node out of range");
  
  return _rca->add_managed_region(region_base, region_length, numa_node);
}

void Rca_AVL::add_managed_region(const std::string& pmem_file, int numa_node)
{
  if(numa_node > Rca::max_numa_node)
    throw std::invalid_argument("numa node out of range");
  
  return _rca->add_managed_region(pmem_file, numa_node);
}

void Rca_AVL::inject_allocation(void * ptr, size_t size, int numa_node)
{
  if(numa_node > Rca::max_numa_node)
    throw std::invalid_argument("numa node out of range");

  // TODO ? space for size  ?
  return _rca->inject_allocation(ptr, size, numa_node);
}

void * Rca_AVL::alloc(size_t size,
                      int numa_node,
                      size_t alignment)
{
  if(size == 0)
    throw std::invalid_argument("invalid size");
  
  if(numa_node > Rca::max_numa_node)
    throw std::invalid_argument("numa node out of range");
    
  return _rca->alloc(size, numa_node, alignment);
}

void Rca_AVL::free(void *ptr, int numa_node)
{
  _rca->free(ptr, numa_node);
}


} // nupm
