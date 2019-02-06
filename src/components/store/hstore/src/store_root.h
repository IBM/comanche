#ifndef _COMANCHE_HSTORE_ROOT_H
#define _COMANCHE_HSTORE_ROOT_H

struct store_root_t
{
  /* A pointer so that null value can indicate no allocation.
   * Locates a pc_al_t.
   * - all allocated space can be accessed through pc
   * If using allocator_cc:
   *   - space controlled by the allocator immediately follows the pc.
   *   - all free space can be accessed through allocator
   */
  PMEMoid persist_oid;
  /* If using allocator_cc, locates a heap_cc, which can be used to construct a allocator_cc
   * If using allocator_co, locates a heap_co, which can be used to construct a allocator_co
   */
  PMEMoid heap_oid;
};

#endif
