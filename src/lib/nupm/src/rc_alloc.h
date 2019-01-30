#ifndef __NUPM_RC_ALLOC__
#define __NUPM_RC_ALLOC__

#include <string>
#include <common/memory.h>

namespace nupm
{
class Rca_AVL_internal;

/** 
 * Reconstituting allocator
 * 
 * 
 */
class Rca_AVL : public Common::Reconstituting_allocator
{
 public:
  Rca_AVL();
  ~Rca_AVL();

  void add_managed_region(void * region_base, size_t region_length, int numa_node);
  void add_managed_region(const std::string& pmem_file, int numa_node);
  
  void inject_allocation(void * ptrm, size_t size, int numa_node) override;

  void *alloc(size_t size,
              int numa_node,
              size_t alignment = 0) override;

  void free(void *ptr, int numa_node) override;
  
 private:
  Rca_AVL_internal * _rca;

};



}
#endif
