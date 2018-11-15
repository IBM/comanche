#include "vmem_numa.h"

namespace nupm
{

Vmem_allocator::Vmem_allocator()
{
    std::vector<std::pair<void *, size_t>> regions(_n_sockets);

    /* determine contiguous regions */
    for(unsigned s=0; s<_n_sockets; s++) {

      unsigned long region_start = 0;
      unsigned long region_end = 0;
      
      for(auto& m: _mappings[s]) {
        PLOG("mapping: %p %lu -> %u", m.first, m.second, s);
        
        if(region_start == 0) {
          region_start = ((unsigned long)m.first);
          region_end = region_start + m.second;
          _vmm_bases[s] = region_start;
        }
        else if(region_end == ((unsigned long)m.first)) {
          region_end += m.second;
        }
        else throw Logic_exception("unexpected condition");

        regions[s] = std::make_pair((void*)region_start, region_end - region_start);

        //        break; // temporary short circuit to just one DIMM 512GB
      }

      _vmm_ends[s] = _vmm_bases[s] + regions[s].second;
    }
    PLOG("region[0]=> %p %lu", regions[0].first, regions[0].second);
    PLOG("region[1]=> %p %lu", regions[1].first, regions[1].second);

    /* create vmem pools */
    for(unsigned s=0; s<_n_sockets; s++) {
      _vmm[s] = vmem_create_in_region(regions[s].first, regions[s].second);
      if(_vmm[s] == nullptr)
        throw General_exception("vmem_create_in_region failed unexpectedly");
      PLOG("created pool: %p size=%lu", _vmm[s], regions[s].second);
    }
}


Vmem_allocator::~Vmem_allocator()
{
  for(unsigned s=0; s<_n_sockets; s++)
    vmem_delete(_vmm[s]);
}


void * Vmem_allocator::alloc(int numa_node, size_t size)
{
  if(numa_node == -1)
    numa_node = 0;

  assert(_vmm[numa_node]);
  auto& vmm = _vmm[numa_node];
  if(vmm == nullptr)
    throw API_exception("invalid numa node (%d)", numa_node);
    
  return vmem_malloc(vmm, size);
}

void Vmem_allocator::free(void * ptr)
{
  vmem_free(vmem_for(ptr), ptr);
}

VMEM * Vmem_allocator::vmem_for(void *ptr) {
  for(unsigned s=0; s<_n_sockets; s++) {
    unsigned long p = ((unsigned long) ptr);
    if(p >= _vmm_bases[s] && p < _vmm_ends[s]) return _vmm[s];
  }
  throw API_exception("cannot resolve pointer to vmm pool");
  return nullptr;
}


  
} // namespace nupm
