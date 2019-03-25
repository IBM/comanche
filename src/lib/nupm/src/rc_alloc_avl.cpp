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


#include <common/exceptions.h>
#include <common/logging.h>
#include <core/avl_malloc.h>
#include <core/slab.h>
#include <numa.h>
#include <stdexcept>

#include "rc_alloc_avl.h"

namespace nupm
{
namespace Rca
{
int max_numa_node;
}

__attribute__((constructor)) static void init_Rca()
{
  Rca::max_numa_node = numa_max_node();
};

class Rca_AVL_internal {
  static constexpr unsigned _debug_level = 1;

 public:
  Rca_AVL_internal() : _slab()
  {
    _allocators.resize(Rca::max_numa_node + 1, 0);
    for (int i = 0; i <= Rca::max_numa_node; i++) _allocators[i] = 0;
  }

  ~Rca_AVL_internal()
  {
    for (auto &i : _allocators) {
      if (i != nullptr) delete i;
    }
  }

  void add_managed_region(void * region_base,
                          size_t region_length,
                          int    numa_node)
  {
    assert(region_base);
    assert(region_length > 0);

    if (_allocators[numa_node] == nullptr) {
      _allocators[numa_node] = new Core::AVL_range_allocator(
          _slab, reinterpret_cast<addr_t>(region_base), region_length);
    }
    else {
      _allocators[numa_node]->add_new_region(
          reinterpret_cast<addr_t>(region_base), region_length);
    }
  }

  void inject_allocation(void *ptr, size_t size, int numa_node)
  {
    assert(ptr);
    assert(_allocators[numa_node]);
    
    auto mrp = _allocators[numa_node]->alloc_at(reinterpret_cast<addr_t>(ptr), size);
    if (mrp == nullptr)
      throw General_exception("alloc_at on AVL range allocator failed unexpectedly");
  }

  void *alloc(size_t size, int numa_node, size_t alignment)
  {
    auto mr = _allocators[numa_node]->alloc(size, alignment);
    if (_debug_level > 1) PLOG("allocated: 0x%lx size=%lu", mr->addr(), size);

    assert(mr);
    return mr->paddr();
  }

  void free(void *ptr, int numa_node)
  {
    if (ptr == nullptr)
      throw API_exception("pointer argument to free cannot be null");

    _allocators[numa_node]->free(reinterpret_cast<addr_t>(ptr));
  }

  void debug_dump(std::string *out_str)
  {
    if(_allocators[0])
      _allocators[0]->dump_info(out_str);

    if(_allocators[1])
    _allocators[1]->dump_info(out_str);
  }

 private:
  Core::Slab::CRuntime<Core::Memory_region> _slab; /* use C runtime for slab? */
  std::vector<Core::AVL_range_allocator *>  _allocators;
};

Rca_AVL::Rca_AVL() : _rca(new Rca_AVL_internal()) {}

Rca_AVL::~Rca_AVL() { delete _rca; }

void Rca_AVL::add_managed_region(void * region_base,
                                 size_t region_length,
                                 int    numa_node)
{
  if (numa_node > Rca::max_numa_node)
    throw std::invalid_argument("numa node out of range");

  _rca->add_managed_region(region_base, region_length, numa_node);
}

void Rca_AVL::inject_allocation(void *ptr, size_t size, int numa_node)
{
  if (numa_node > Rca::max_numa_node)
    throw std::invalid_argument("numa node out of range");

  _rca->inject_allocation(ptr, size, numa_node);
}

void *Rca_AVL::alloc(size_t size, int numa_node, size_t alignment)
{
  if (size == 0) throw std::invalid_argument("invalid size");
  if (alignment == 0) alignment = 1;

  if (numa_node > Rca::max_numa_node)
    throw std::invalid_argument("numa node out of range");

  return _rca->alloc(size, numa_node, alignment);
}

void Rca_AVL::free(void *ptr, int numa_node, size_t size)
{
  _rca->free(ptr, numa_node);
}

void Rca_AVL::debug_dump(std::string *out_log)
{
  _rca->debug_dump(out_log);
}

}  // namespace nupm
