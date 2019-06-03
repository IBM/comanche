/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include "persist_session.h"

using namespace nvmestore;

status_t persist_session::may_ajust_io_mem(size_t value_len)
{
  /*  increase IO buffer sizes when value size is large*/
  // TODO: need lock
  if (value_len > _io_mem_size) {
    size_t new_io_mem_size = _io_mem_size;

    while (new_io_mem_size < value_len) {
      new_io_mem_size *= 2;
    }

    _io_mem_size = new_io_mem_size;
    _blk_manager->free_io_buffer(_io_mem);

    _io_mem = _blk_manager->allocate_io_buffer(_io_mem_size, 4096,
                                               Component::NUMA_NODE_ANY);
    if (option_DEBUG)
      PINF("[Nvmestore_session]: incresing IO mem size %lu at %lx",
           new_io_mem_size, _io_mem);
  }
  return S_OK;
}
