#ifndef __NUPM_RC_ALLOC__
#define __NUPM_RC_ALLOC__

#include <string>
#include <common/memory.h>


namespace nupm
{
class Rca_AVL_internal;

/** 
 * Reconstituting allocator.  Metadata is held in DRAM (C runtime allocator).
 * 
 * 
 */
class Rca_AVL : public Common::Reconstituting_allocator
{
 public:
  Rca_AVL();
  ~Rca_AVL();

  /** 
   * Add region of memory to be managed
   * 
   * @param region_base Base of region
   * @param region_length Size of region in bytes
   * @param numa_node NUMA node
   */
  void add_managed_region(void * region_base, size_t region_length, int numa_node);

  /** 
   * EXPERIMENTAL: for memkind testing only
   * 
   * @param pmem_file Persistent memory filename (e.g., /mnt/pmem0)
   * @param numa_node NUMA node
   */
  void add_managed_region(const std::string& pmem_file, int numa_node);

  /** 
   * Allocate region of memory
   * 
   * @param size Size of memory in bytes
   * @param numa_node NUMA node
   * @param alignment Required alignment
   * 
   * @return Pointer to newly allocated region
   */
  void *alloc(size_t size,
              int numa_node,
              size_t alignment = 0) override;

  /** 
   * Free previously allocated region of memory
   * 
   * @param ptr Point to region
   * @param numa_node NUMA node
   */
  void free(void *ptr, int numa_node) override;

  /** 
   * Reconstitute a previous allocation.  Mark memory as allocated.
   * 
   * @param p Address of region
   * @param size Size of region in bytes
   * @param numa_node NUMA node
   */
  void inject_allocation(void * p, size_t size, int numa_node) override;

  
 private:
  Rca_AVL_internal * _rca;

};



}
#endif
