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

#if 0
/** Allocate space for metadata
 *
 * Before invoke this, we ensure that the key is either not exsiting before, or
 * have been freed
 */
void persist_session::alloc_new_object(const std::string& key,
                                       size_t             value_len,
                                       obj_info_t*&       out_blkmeta)
{
  auto& blk_manager = this->_blk_manager;

  uint64_t hashkey = CityHash64(key.c_str(), key.length());

  size_t blk_size     = blk_manager->blk_sz();
  size_t nr_io_blocks = (value_len + blk_size - 1) / blk_size;

  void* handle;

  // transaction also happens in here
  uint64_t lba = _blk_manager->alloc_blk_region(nr_io_blocks, &handle);

  PDBG("write to lba %lu with length %lu, key %lx", lba, value_len, hashkey);

  auto& objinfo = out_blkmeta;
  TX_BEGIN(pop)
  {
    /* allocate memory for entry - range added to tx implicitly? */

    // get the available range from allocator
    objinfo = TX_ALLOC(struct obj_info, sizeof(struct obj_info));

    D_RW(objinfo)->lba_start = lba;
    D_RW(objinfo)->size      = value_len;
    D_RW(objinfo)->handle    = handle;
    D_RW(objinfo)->key_len   = key.length();
    TOID(char) key_data      = TX_ALLOC(char, key.length() + 1);  // \0 included

    if (D_RO(key_data) == nullptr)
      throw General_exception("Failed to allocate space for key");
    std::copy(key.c_str(), key.c_str() + key.length() + 1, D_RW(key_data));

    D_RW(objinfo)->key_data = key_data;

    /* insert into HT */
    int rc;
    if ((rc = hm_tx_insert(pop, D_RW(root)->map, hashkey, objinfo.oid))) {
      if (rc == 1)
        throw General_exception("inserting same key");
      else
        throw General_exception("hm_tx_insert failed unexpectedly (rc=%d)", rc);
    }

    _num_objs += 1;
  }
  TX_ONABORT
  {
    // TODO: free objinfo
    throw General_exception("TX abort (%s) during nvmeput", pmemobj_errormsg());
  }
  TX_END
  PDBG("Allocated obj with obj %p, ,handle %p", D_RO(objinfo),
       D_RO(objinfo)->handle);
}
#endif

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

#if 0
status_t persist_session::put(const std::string& key,
                              const void*        valude,
                              size_t             value_len,
                              unsigned int       flags)
{
  uint64_t hashkey = CityHash64(key.c_str(), key.length());
  size_t   blk_sz  = _blk_manager->blk_sz();

  struct obj_info* blkmeta;  // block mapping of this obj

  if (hm_tx_lookup(pop, D_RO(root)->map, hashkey)) {
  }

  if (try_get(key) == S_OK) {  // key exsits
    PLOG("overriting exsiting obj");
    erase(key);
    return put(key, value, value_len, flags);
  }

  may_ajust_io_mem(value_len);

  alloc_new_object(key, value_len, blkmeta);
  if (blkmeta == nullptr) {  // key exist before
    erase(key);
    return put(key, value, value_len, flags);
  }

  memcpy(_blk_manager->virt_addr(_io_mem), value,
         value_len); /* for the moment we have to memcpy */

#ifdef USE_ASYNC
#error("use_sync is deprecated")
  // TODO: can the free be triggered by callback?
  uint64_t tag = blk_dev->async_write(session->io_mem, 0, lba, nr_io_blocks);
  D_RW(objinfo)->last_tag = tag;
#else
  auto nr_io_blocks = (value_len + blk_sz - 1) / blk_sz;
  _blk_manager->do_block_io(nvmestore::BLOCK_IO_WRITE, _io_mem,
                            D_RO(blkmeta)->lba_start, nr_io_blocks);
#endif
  return S_OK;
}
#endif

status_t persist_session::map_keys(std::function<int(const std::string& key)> f)
{
  _meta_store->map_keys(_meta_pool, f);
  return S_OK;
}
