/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include <api/block_allocator_itf.h>
#include <api/block_itf.h>
#include <api/kvstore_itf.h>
#include <common/rand.h>
#include "bitmap_ikv.h"

using namespace Component;
namespace block_alloc_ikv
{
static unsigned size_to_bin(uint64_t size)
{
  return 64 - __builtin_clzll(size);
}

class BlockAlloc_ikv {
 public:
  /**
   * Construct a bitmap in a IKVstore pool
   * @param store
   * @param pool
   * @param nbits number of
   */
  BlockAlloc_ikv(const std::string& id,
                 IKVStore*          store,
                 IKVStore::pool_t   pool,
                 unsigned           nbits)
      : _id(id), _store(store), _pool(pool)
  {
    _bitmap = new bitmap_ikv(_store, _pool, _id);
    if (nbits > _bitmap->get_capacity())
      throw General_exception("single chunk is too small");
  }

  ~BlockAlloc_ikv() { delete _bitmap; };

  lba_t alloc(size_t size, void** handle);
  void  free(lba_t addr, void* handle);

 private:
  IKVStore*        _store;
  IKVStore::pool_t _pool;
  std::string      _id; /** identifier for this allocator/block device*/
  bitmap_ikv*      _bitmap;
};

lba_t BlockAlloc_ikv::alloc(size_t count, void** handle)
{
  assert(handle);
  lba_t  ret_pos;
  size_t order = size_to_bin(count) - 1;

  IKVStore::key_t lockkey;
  _bitmap->load(lockkey);
  // TODO sth bad can happen here
  ret_pos = _bitmap->find_free_region(order);
  _bitmap->flush(lockkey);

  return ret_pos;
}

lba_t free(lba_t addr, void* handle) { assert(handle); }
}  // namespace block_alloc_ikv
