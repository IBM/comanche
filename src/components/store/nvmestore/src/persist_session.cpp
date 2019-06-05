/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include <city.h>
#include <set>
#include "persist_session.h"

using namespace nvmestore;

/** Allocate space for metadata
 *
 * Before invoke this, we ensure that the key is either not exsiting before, or
 * have been freed
 */
void persist_session::alloc_new_object(const std::string& key,
                                       size_t             value_len,
                                       obj_info_t*&       out_objinfo)
{
  auto& blk_manager = this->_blk_manager;

  size_t blk_size     = blk_manager->blk_sz();
  size_t nr_io_blocks = (value_len + blk_size - 1) / blk_size;

  void* handle;

  // get free block range
  uint64_t lba = _blk_manager->alloc_blk_region(nr_io_blocks, &handle);
  PDBG("write to lba %lu with length %lu, key %lx", lba, value_len, hashkey);

  // prepare objinfo buffer
  void* raw_objinfo = malloc(sizeof(obj_info_t));
  if (raw_objinfo == nullptr)
    throw General_exception("Failed to allocate space for objinfo");
  obj_info_t* objinfo = reinterpret_cast<obj_info_t*>(raw_objinfo);

  // compose the objinfo
  objinfo->lba_start = lba;
  objinfo->size      = value_len;
  objinfo->handle    = handle;
  objinfo->key_len   = key.length();

  void* key_data = malloc(key.length() + 1);  // \0 included
  if (key_data == nullptr)
    throw General_exception("Failed to allocate space for key");
  std::copy(key.c_str(), key.c_str() + key.length() + 1, (char*) (key_data));
  objinfo->key_data = (char*) key_data;

  // put to meta_pool
  status_t rc =
      _meta_store->put(_meta_pool, key, raw_objinfo, sizeof(obj_info_t));
  if (rc != S_OK) {
    throw General_exception("put objinfo to meta_pool failed");
  }

  _num_objs += 1;

  PDBG("Allocated obj with obj %p, ,handle %p", D_RO(objinfo),
       D_RO(objinfo)->handle);

  out_objinfo = objinfo;
}

status_t persist_session::erase(const std::string& key)
{
  if (check_exists(key) == E_NOT_FOUND) return IKVStore::E_KEY_NOT_FOUND;

  uint64_t pool = reinterpret_cast<uint64_t>(this);

  status_t rc;
  void*    raw_objinfo;
  size_t   obj_info_sz_in_bytes;

  // Get objinfo from meta_pool
  rc = _meta_store->get(_meta_pool, key, raw_objinfo, obj_info_sz_in_bytes);
  if (rc != S_OK || obj_info_sz_in_bytes != sizeof(obj_info_t)) {
    throw General_exception("get objinfo from meta_pool failed");
  }
  obj_info_t* objinfo = reinterpret_cast<obj_info_t*>(raw_objinfo);

  /* get hold of write lock to remove */
  if (!p_state_map->state_get_write_lock(pool, objinfo->handle))
    throw API_exception("unable to remove, value locked");

  PDBG("Tring to Remove obj with obj %p,handle %p", D_RO(objinfo),
       D_RO(objinfo)->handle);

  // Free objinfo from meta_pool
  rc = _meta_store->erase(_meta_pool, key);
  if (rc != S_OK) {
    throw General_exception("erase objinfo from meta_pool failed");
  }

  // Free block range in the blk_alloc
  _blk_manager->free_blk_region(objinfo->lba_start, objinfo->handle);
  p_state_map->state_remove(pool, objinfo->handle);

  if (objinfo->key_data) free(objinfo->key_data);
  if (objinfo) free(objinfo);

  _num_objs -= 1;
  return S_OK;
}

/*
 * Search whether key exists in this pool or not
 *
 * @return S_OK if exists, otherwise E_NOT_FOUND
 * Use uint64_64_t for fast searching
 */
status_t persist_session::check_exists(const std::string in_key) const
{
  std::set<uint64_t> all_hashkeys;
  _meta_store->map_keys(
      _meta_pool, [&all_hashkeys](const std::string& key) -> int {
        uint64_t hashkey = CityHash64(key.c_str(), key.length());
        all_hashkeys.insert(hashkey);
        return 0;
      });
  if (all_hashkeys.find(CityHash64(in_key.c_str(), in_key.length())) ==
      all_hashkeys.end()) {
    return E_NOT_FOUND;
  }
  return S_OK;
}

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

status_t persist_session::get(const std::string& key,
                              void*&             out_value,
                              size_t&            out_value_len)
{
  if (check_exists(key) == E_NOT_FOUND) return IKVStore::E_KEY_NOT_FOUND;
  size_t blk_sz = _blk_manager->blk_sz();

  void*  objinfo;
  size_t obj_info_length;

  _meta_store->get(_meta_pool, key, objinfo, obj_info_length);

  auto val_len = ((obj_info_t*) (objinfo))->size;
  auto lba     = ((obj_info_t*) (objinfo))->lba_start;

#ifdef USE_ASYNC
  uint64_t tag = D_RO(objinfo)->last_tag;
  while (!blk_dev->check_completion(tag))
    cpu_relax(); /* check the last completion, TODO: check each time makes the
                    get slightly slow () */
#endif
  PDBG("prepare to read lba %d with length %d, key %lx", lba, val_len, hashkey);

  may_ajust_io_mem(val_len);
  size_t nr_io_blocks = (val_len + blk_sz - 1) / blk_sz;

  _blk_manager->do_block_io(nvmestore::BLOCK_IO_READ, _io_mem, lba,
                            nr_io_blocks);

  out_value = malloc(val_len);
  assert(out_value);
  memcpy(out_value, _blk_manager->virt_addr(_io_mem), val_len);
  out_value_len = val_len;

  return S_OK;
}

status_t persist_session::put(const std::string& key,
                              const void*        value,
                              size_t             value_len,
                              unsigned int       flags)
{
  if (check_exists(key) == S_OK) {
#if 0
    erase(key);
    return put(key, value, value_len, flags);
#endif
    throw API_exception("put overwrite not implemented, before erase");
  }
  size_t blk_sz = _blk_manager->blk_sz();

  obj_info_t* objinfo;  // block mapping of this obj

  may_ajust_io_mem(value_len);

  alloc_new_object(key, value_len, objinfo);
  auto lba = ((obj_info_t*) (objinfo))->lba_start;
  memcpy(_blk_manager->virt_addr(_io_mem), value,
         value_len); /* for the moment we have to memcpy */

#ifdef USE_ASYNC
#error("use_sync is deprecated")
  // TODO: can the free be triggered by callback?
  uint64_t tag = blk_dev->async_write(session->io_mem, 0, lba, nr_io_blocks);
  D_RW(objinfo)->last_tag = tag;
#else
  auto nr_io_blocks = (value_len + blk_sz - 1) / blk_sz;
  _blk_manager->do_block_io(nvmestore::BLOCK_IO_WRITE, _io_mem, lba,
                            nr_io_blocks);
#endif
  return S_OK;
}

status_t persist_session::map_keys(std::function<int(const std::string& key)> f)
{
  _meta_store->map_keys(_meta_pool, f);
  return S_OK;
}
