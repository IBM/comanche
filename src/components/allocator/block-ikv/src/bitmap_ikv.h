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

#ifndef BITMAP_IKV_H_
#define BITMAP_IKV_H_

#include <api/kvstore_itf.h>
#include <component/base.h>
#include "cstring"
/* Note: these uuid decls are so we don't have to have access to the component
 * source code */

#ifndef MB
#define MB(x) (x << 20)
#endif

#define BITS_PER_LONG (64)
#define BITS_TO_LONGS(nr) (((nr) + (BITS_PER_LONG - 1)) / (BITS_PER_LONG))

using namespace Component;

namespace block_alloc_ikv
{
using word_t = uint64_t;  // bitmap operation is done in word
enum reg_op_t {
  REG_OP_ISFREE,   // region is all zero bits
  REG_OP_ALLOC,    // set all bits in this region
  REG_OP_RELEASE,  // clear all bits in region
};

/**
 * Fixed sized bitmap(4M bits)
 *
 * Each instance is backed by one object in the ikvstore pool
 */
class bitmap_ikv {
 public:
  static constexpr size_t BITMAP_CHUNCK_SIZE = MB(4); /** intial bitmap*/

  /**
   * construct a bitmap in a IKVstore pool
   * @param store
   * @param pool
   * @param nbits number of
   */
  bitmap_ikv(IKVStore* store, IKVStore::pool_t pool, const std::string& id)
      : _store(store), _pool(pool), _id(id)
  {
    _capacity = BITMAP_CHUNCK_SIZE * 8;
  }

  ~bitmap_ikv(){};

  /** Load bitmap from ikvstore*/
  status_t load(IKVStore::key_t& out_lockkey)
  {
    void*           ptr2bitmap;
    IKVStore::key_t lock_key;
    size_t          bitmap_len = BITMAP_CHUNCK_SIZE;

    if (S_OK != _store->lock(_pool, _id, IKVStore::STORE_LOCK_WRITE, ptr2bitmap,
                             bitmap_len, lock_key)) {
      throw General_exception("bitmap_ikv: intial kv lock failed");
    }
    out_lockkey = lock_key;
    _bitdata    = reinterpret_cast<word_t*>(ptr2bitmap);
    return S_OK;
  }

  /** Flush this bitmap chunk to ikvstore*/
  status_t flush(IKVStore::key_t lockkey)
  {
    if (S_OK != _store->unlock(_pool, lockkey))
      throw General_exception("bitmap_ikv: unlock failed");
    return S_OK;
  }

  status_t zero()
  {
    size_t mem_size = BITS_TO_LONGS(_capacity) * sizeof(long);
    memset(_bitdata, 0, mem_size);
    return S_OK;
  }

  /*
   * find a contiguous aligned mem region, this assume data is loaded
   *
   * @param pop persist object pool
   * @param bitmap the bitmap to operate on
   * @param order, the order(2^order free blocks) to set from 0 to 1
   */
  int find_free_region(unsigned order);

  /*
   * free a region from bitmap, assuming data is loaded
   *
   * @param pos the starting bit to reset
   * @param order, the order(2^order bits) to reset
   */
  int release_region(unsigned int pos, unsigned order);

  /**
   * Get the capacity in bits
   */
  size_t get_capacity() const { return _capacity; }

 private:
  int _reg_op(unsigned int pos, unsigned order, reg_op_t reg_op);

  IKVStore*        _store;
  IKVStore::pool_t _pool;
  std::string      _id; /** identifier for this allocator/block device*/

  size_t  _capacity; /** how many bit in this bitmap*/
  word_t* _bitdata;
};  // namespace block_alloc_ikv

}  // namespace block_alloc_ikv

#endif
