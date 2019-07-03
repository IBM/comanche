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

#define GET_ALLOC_LBA(handle) (((uint64_t) handle) >> 32)  // higher 32bit
#define GET_ALLOC_ORDER(handle) \
  (((uint64_t) handle) & 0xffffffff)  // lower 32bit

using namespace Component;
namespace block_alloc_ikv
{
static unsigned size_to_bin(uint64_t size)
{
  return 64 - __builtin_clzll(size);
}

class BlockAlloc_ikv : public Component::IBlock_allocator {
 public:
  /**
   * Construct a bitmap in a IKVstore pool
   * @param store
   * @param pool
   * @param nbits number of
   */
  BlockAlloc_ikv(size_t             nbits,
                 IKVStore*          store,
                 IKVStore::pool_t   pool,
                 const std::string& id,
                 bool               force_init)
      : _store(store), _pool(pool), _id(id)
  {
    _bitmap = new bitmap_ikv(_store, _pool, _id);
    if (nbits > _bitmap->get_capacity())
      throw General_exception("single chunk is too small");

    if (force_init) {
      IKVStore::key_t lockkey;
      _bitmap->load(lockkey);
      _bitmap->zero();
      _bitmap->flush(lockkey);
    }
  }

  virtual ~BlockAlloc_ikv() { delete _bitmap; };

  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x7f23a369,
                         0x1993,
                         0x488b,
                         0x95cf,
                         0x8c,
                         0x77,
                         0x3c,
                         0xc5,
                         0xaa,
                         0xf3);

  void* query_interface(Component::uuid_t& itf_uuid) override
  {
    if (itf_uuid == Component::IBlock_allocator::iid()) {
      return (void*) static_cast<Component::IBlock_allocator*>(this);
    }
    else
      return NULL;  // we don't support this interface
  }

  void unload() override { delete this; }

  virtual lba_t    alloc(size_t size, void** handle) override;
  virtual void     free(lba_t addr, void* handle) override;
  virtual status_t resize(lba_t addr, size_t size) override;
  virtual size_t   get_free_capacity() override;
  virtual size_t   get_capacity() override;

 private:
  IKVStore*        _store;
  IKVStore::pool_t _pool;
  std::string      _id; /** identifier for this allocator/block device*/
  bitmap_ikv*      _bitmap;
};

/**
 * Allocate a block range
 *
 * We assume (lba->block_range) mapping is made persist outside this component
 *(e.g. nvmestore.) So that during restart,  the exact same handle is passed in.
 **/
lba_t BlockAlloc_ikv::alloc(size_t count, void** handle)
{
  assert(handle);
  lba_t    ret_pos;
  unsigned order = size_to_bin(count) - 1;

  IKVStore::key_t lockkey;
  _bitmap->load(lockkey);
  // TODO sth bad can happen here
  ret_pos = _bitmap->find_free_region(order);
  _bitmap->flush(lockkey);

  // form a void *
  *handle = (void*) (ret_pos << 32 | order);
  return ret_pos;
}

/**
 * Free a block range
 *
 * We assume (lba->block_range) mapping is made persist outside this component
 *(e.g. nvmestore.) So that during restart,  the exact same handle is passed in.
 **/
void BlockAlloc_ikv::free(lba_t addr, void* handle)
{
  unsigned lba_start = GET_ALLOC_LBA(handle);
  unsigned order     = GET_ALLOC_ORDER(handle);
  if (handle == nullptr || lba_start != addr) {
    throw General_exception(
        "[%s]: try to free block region using corrupted handle", __func__);
  }

  IKVStore::key_t lockkey;
  _bitmap->load(lockkey);
  _bitmap->release_region(lba_start, order);
  _bitmap->flush(lockkey);
}

size_t   BlockAlloc_ikv::get_free_capacity() { return 0; }
size_t   BlockAlloc_ikv::get_capacity() { return 0; }
status_t BlockAlloc_ikv::resize(lba_t addr, size_t size) { return E_NOT_IMPL; }

class BlockAlloc_ikv_factory : public Component::IBlock_allocator_factory {
 public:
  /**
   * Component/interface management
   *
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac3a369,
                         0x1993,
                         0x488b,
                         0x95cf,
                         0x8c,
                         0x77,
                         0x3c,
                         0xc5,
                         0xaa,
                         0xf3);

  void* query_interface(Component::uuid_t& itf_uuid) override
  {
    if (itf_uuid == Component::IBlock_allocator_factory::iid()) {
      return (void*) static_cast<Component::IBlock_allocator_factory*>(this);
    }
    else
      return NULL;  // we don't support this interface
  }

  void unload() override { delete this; }

  /**
   * Open an allocator
   *
   * @return Pointer to allocator instance. Ref count = 1. Release ref to
   * delete.
   */

  virtual Component::IBlock_allocator* open_allocator(size_t           max_lba,
                                                      IKVStore*        store,
                                                      IKVStore::pool_t pool,
                                                      std::string&     name,
                                                      bool force_init) override
  {
    {
      Component::IBlock_allocator* obj =
          static_cast<Component::IBlock_allocator*>(
              new BlockAlloc_ikv(max_lba, store, pool, name, force_init));

      obj->add_ref();
      return obj;
    }
    return NULL;
  }
};

}  // namespace block_alloc_ikv

extern "C" void* factory_createInstance(Component::uuid_t& component_id)
{
  if (component_id == block_alloc_ikv::BlockAlloc_ikv_factory::component_id()) {
    return static_cast<void*>(new block_alloc_ikv::BlockAlloc_ikv_factory());
  }
  else
    return NULL;
}
