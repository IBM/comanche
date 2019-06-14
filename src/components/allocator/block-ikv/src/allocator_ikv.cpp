/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include <api/block_allocator_itf.h>
#include <api/block_itf.h>
#include <api/kvstore_itf.h>
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
   * construct a bitmap in a IKVstore pool
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
  }

  ~BlockAlloc_ikv();

  lba_t alloc(size_t size, void** handle);
  lba_t free(lba_t addr, void* handle);

 private:
  IKVStore*        _store;
  IKVStore::pool_t _pool;
  std::string      _id; /** identifier for this allocator/block device*/
  bitmap_ikv*      _bitmap;
};

lba_t BlockAlloc_ikv::alloc(size_t count, void** handle)
{
  int             ret_pos;
  void*           ptr2bitmap;
  IKVStore::key_t lock_key;
  size_t          bitmap_len = bitmap_ikv::BITMAP_CHUNCK_SIZE;

  if (S_OK != _store->lock(_pool, _id, IKVStore::STORE_LOCK_WRITE, ptr2bitmap,
                           bitmap_len, lock_key)) {
    throw General_exception("bitmap_ikv: intial kv lock failed");
  }

  unsigned order = size_to_bin(count) - 1;

  ret_pos = _bitmap->find_free_region(order);

  if (S_OK != _store->unlock(_pool, lock_key)) {
    throw General_exception("bitmap_ikv: unlock failed");
  }

  return ret_pos;
}

}  // namespace block_alloc_ikv
