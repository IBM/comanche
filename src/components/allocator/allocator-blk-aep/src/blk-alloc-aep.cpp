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

/*
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 *
 */

#include <assert.h>
#include <core/physical_memory.h>
#include <libpmemobj.h>
#include <libpmempool.h>
#include <thread>
#include "blk-alloc-aep.h"

#include <boost/filesystem.hpp>

#define RESET_STATE  // testing only

using namespace Component;

static std::mutex _mutex;

static unsigned size_to_bin(uint64_t size)
{
  return 64 - __builtin_clzll(size);
}

static size_t bin_to_size(unsigned bin) { return 1 << bin; }

/*
 * a record of a continuous allocation in the bitmap
 */
struct BlockAllocRecord {
  lba_t    _offset;  // start offset of the allocation
  unsigned _order;   // allocation size is 2^order
};

Block_allocator_AEP::Block_allocator_AEP(size_t      max_lba,
                                         std::string path,
                                         std::string name,
                                         int         numa_node,
                                         bool        force_init)
{
  char const *        pool_name;
  static PMEMobjpool *pop;
  size_t              est_map_size;  // actual size of the pool

  assert(max_lba > 0);
  _nbits = max_lba;

  // size_t est_map_size = BITS_TO_LONGS(_nbits)*sizeof(long) + sizeof(struct
  // bitmap_tx);
  est_map_size = MB(200);

  std::string fullpath;
  if (path[path.length() - 1] != '/')
    fullpath = path + "/" + name;
  else
    fullpath = path + name;

  pool_name = fullpath.c_str();

  PINF("Block_allocator_AEP: poolpath: %s max_lba=%lu", pool_name, max_lba);

  if (access(fullpath.c_str(), F_OK) != 0) {
    PLOG("Creating new Pool: %s", name.c_str());

    boost::filesystem::path p(fullpath);
    boost::filesystem::create_directories(p.parent_path());

    // open an object pool
    pop = pmemobj_create(pool_name, POBJ_LAYOUT_NAME(bitmap_store),
                         est_map_size, 0666);
    if (not pop)
      throw General_exception("%s: failed to create new pool for bitmap - %s\n",
                              __func__, pmemobj_errormsg());
  }
  else {
    PINF("%s: using existing pool", __func__);
    pop = pmemobj_open(pool_name, POBJ_LAYOUT_NAME(bitmap_store));

    // the same pool already exsit?
    if (pop == NULL)
      throw General_exception("failed to re-open pool - %s\n",
                              pmemobj_errormsg());
  }

  // open the bitmap, currently the pool only stores 1 map
  _map = POBJ_ROOT(pop, struct bitmap_tx);

  if (D_RO(_map)->nbits == 0) {
    PINF("%s: create bitmap", __func__);
    if (bitmap_tx_create(pop, _map, _nbits)) {
      throw General_exception("%s, create map failed %s\n", __func__,
                              pmemobj_errormsg());
    }
  }
  else {
    PINF("%s: using existing map", __func__);
  }

  _pop       = pop;
  _pool_name = pool_name;
}

Block_allocator_AEP::~Block_allocator_AEP()
{
#if 0
  PINF("%s: deleting Allocator", __func__);
  // TODO release the map?
  if(bitmap_tx_release(_pop, _map)){
      throw General_exception("failed to release bitmap - %s\n", pmemobj_errormsg());
      }
  PINF("%s: bitmap released", __func__);
#endif
  if (_pop) {
    pmemobj_close(_pop);
    PINF("%s: block-alloc-aep: pmemobj closed", __func__);
  }
}

/* IBlock_allocator */

/**
 * Allocate N contiguous blocks
 *
 * @param size Number of blocks to allocate
 *
 * @return Logical block address of start of allocation. Throw exception on
 * insufficient blocks.
 */
lba_t Block_allocator_AEP::alloc(size_t count, void **handle)
{
  lba_t    pos;    // the starting addr of the allocated addr
  unsigned order;  // order of the allocation size
  int      ret = -1;

  order = size_to_bin(count) - 1;

  pos = bitmap_tx_find_free_region(_pop, _map, order);

  if (pos == -1) {
    throw API_exception("%s: cannot find avail blk with order %lu and nbits "
                        "%lu!, bitmap  map already full",
                        __func__, order, _nbits);
  }

  PDBG("%s: (%lu) find free region at lba %lu, order= %u", __func__,
       std::hash<std::thread::id>{}(std::this_thread::get_id()), pos, order);

  // Handle needs to persist so that we can free this blkrange when restarting
  // app
  TX_BEGIN(_pop)
  {
    TOID(struct BlockAllocRecord) record;
    record = TX_ALLOC(struct BlockAllocRecord, sizeof(struct BlockAllocRecord));
    D_RW(record)->_offset = pos;
    D_RW(record)->_order  = order;

    *handle = reinterpret_cast<void *>(record.oid.off);
  }
  TX_ONABORT
  {
    throw General_exception("TX abort (%s) when allocate block range record",
                            pmemobj_errormsg());
  }
  TX_END
  return pos;
}

/**
 * Free a previous allocation
 *
 * @param addr Logical block address of allocation
 */
void Block_allocator_AEP::free(lba_t lba, void *handle)
{
  PMEMoid oid;
  oid.pool_uuid_lo = _map.oid.pool_uuid_lo;
  oid.off          = reinterpret_cast<uint64_t>(handle);

  if (OID_IS_NULL(oid)) {
    throw General_exception("Double free on persist handle for block range");
  }

  TOID(struct BlockAllocRecord) persist_record;
  persist_record.oid = oid;

  const BlockAllocRecord *rec = D_RO(persist_record);
  PDBG("free previous block range: from %lu, order %u", rec->_offset,
       rec->_order);

  // TODO actually free here
  bitmap_tx_release_region(_pop, _map, rec->_offset, rec->_order);

  TX_BEGIN(_pop) { TX_FREE(persist_record); }
  TX_ONABORT
  {
    throw General_exception("TX abort (%s) when free block range record",
                            pmemobj_errormsg());
  }
  TX_END
}

/* explicitly resize to 0 will remove this allocator*/
status_t Block_allocator_AEP::resize(lba_t addr, size_t size)
{
  status_t ret = -1;
  if (addr == 0 && size == 0) {
    // free bitmap data
    if (bitmap_tx_destroy(_pop, _map)) {
      PERR("%s failed to destroy bitmap ", __func__);
      return ret;
    }

    pmemobj_close(_pop);
    _pop = NULL;

    if (pmempool_rm(_pool_name.c_str(), 0)) {
      throw General_exception("%s unable to delete bitmap pool %s", __func__,
                              _pool_name.c_str());
    }

    PINF("%s allocation info totally removed ", __func__);
  }
  return E_FAIL;
}

/**
 * Get number of free units
 *
 *
 * @return Free capacity in units
 */
size_t Block_allocator_AEP::get_free_capacity() { return 0; }

/**
 * Get total capacity
 *
 *
 * @return Capacity in units
 */
size_t Block_allocator_AEP::get_capacity() { return 0; }

void Block_allocator_AEP::dump_info() {}

/**
 * Factory entry point
 *
 */
extern "C" void *factory_createInstance(Component::uuid_t &component_id)
{
  if (component_id == Block_allocator_AEP_factory::component_id()) {
    return static_cast<void *>(new Block_allocator_AEP_factory());
  }
  else
    return NULL;
}

#undef RESET_STATE
