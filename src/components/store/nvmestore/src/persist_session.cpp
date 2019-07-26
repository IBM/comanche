/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#include <city.h>
#include <set>
#include "persist_session.h"
#include <sys/mman.h>
extern "C" {
#include <spdk/env.h>
}

namespace nvmestore
{
/** Allocate space for metadata
 *
 * Before invoke this, we ensure that the key is either not exsiting before, or
 * have been freed
 */
status_t persist_session::alloc_new_object(const std::string& key,
                                           size_t             value_len,
                                           obj_info_t*&       out_objinfo)
{
  auto& blk_manager = this->_blk_manager;

  size_t blk_size     = blk_manager->blk_sz();
  size_t nr_io_blocks = (value_len + blk_size - 1) / blk_size;

  // prepare objinfo buffer
  size_t obj_info_sz_in_bytes =
      sizeof(obj_info_t) + key.length() + 1;  // \0 included
  void* raw_objinfo = malloc(obj_info_sz_in_bytes);
  memset(raw_objinfo, 0, obj_info_sz_in_bytes);
  if (raw_objinfo == nullptr)
    throw General_exception("Failed to allocate space for objinfo");
  obj_info_t* objinfo = reinterpret_cast<obj_info_t*>(raw_objinfo);

  // compose the objinfo
  // get free block range
  uint64_t lba =
      _blk_manager->alloc_blk_region(nr_io_blocks, &(objinfo->block_region));
  // PDBG("write to lba %lu with length %lu, key %lx", lba, value_len, hashkey);

  objinfo->size      = value_len;
  objinfo->lba_start = lba;
  objinfo->key_len   = key.length();

  void* key_data = (char*) raw_objinfo + sizeof(obj_info_t);
  std::copy(key.c_str(), key.c_str() + key.length() + 1, (char*) (key_data));
  objinfo->key_data = (char*) key_data;

  // put to meta_pool
  status_t rc =
      _meta_store->put(_meta_pool, key, raw_objinfo, obj_info_sz_in_bytes);
  if (rc != S_OK) {
    throw General_exception("put objinfo to meta_pool failed");
  }

  _num_objs += 1;

  PDBG("Allocated obj with obj %p", objinfo);

  out_objinfo = objinfo;
  return S_OK;
}

status_t persist_session::erase(const std::string& key)
{
  if (check_exists(key) == E_NOT_FOUND) return IKVStore::E_KEY_NOT_FOUND;

  uint64_t pool = reinterpret_cast<uint64_t>(this);

  status_t rc;
  void*    raw_objinfo = nullptr;
  size_t   obj_info_sz_in_bytes;

  // Get objinfo from meta_pool
  rc = _meta_store->get(_meta_pool, key, raw_objinfo, obj_info_sz_in_bytes);
  if (rc != S_OK || obj_info_sz_in_bytes < sizeof(obj_info_t)) {
    throw General_exception("get objinfo from meta_pool failed");
  }
  obj_info_t* objinfo = reinterpret_cast<obj_info_t*>(raw_objinfo);

  /* get hold of write lock to remove */
  if (!p_state_map->state_get_write_lock(pool, objinfo->block_region))
    throw API_exception("unable to remove, value locked");

  PDBG("Tring to Remove obj with obj %p",objinfo);

  // Free objinfo from meta_pool
  rc = _meta_store->erase(_meta_pool, key);
  if (rc != S_OK) {
    throw General_exception("erase objinfo from meta_pool failed");
  }

  // Free block range in the blk_alloc
  uint64_t block_region = (uint64_t)(objinfo->block_region);
  unsigned lba_start    = objinfo->lba_start;

  _blk_manager->free_blk_region(lba_start, objinfo->block_region);
  p_state_map->state_remove(pool, objinfo->block_region);

  _meta_store->free_memory(raw_objinfo);

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
#if 1
  void*  raw_objinfo = nullptr;
  size_t obj_info_length;

  status_t rc = _meta_store->get(_meta_pool, in_key, raw_objinfo, obj_info_length);
  if(raw_objinfo){
    _meta_store->free_memory(raw_objinfo);
  }
  return (rc == IKVStore::E_KEY_NOT_FOUND) ? E_NOT_FOUND : S_OK;
#else
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
#endif
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

  void*  raw_objinfo = nullptr;
  size_t obj_info_length;

  status_t rc = _meta_store->get(_meta_pool, key, raw_objinfo, obj_info_length);
  if (rc != S_OK || obj_info_length < sizeof(obj_info_t)) {
    throw General_exception("get objinfo from meta_pool failed");
  }

  obj_info_t* objinfo = reinterpret_cast<obj_info_t*>(raw_objinfo);
  auto        val_len = objinfo->size;
  auto        lba     = objinfo->lba_start;

  PDBG("prepare to read lba 0x%lx with length %d", lba, val_len);

  may_ajust_io_mem(val_len);
  size_t nr_io_blocks = (val_len + blk_sz - 1) / blk_sz;

  _blk_manager->do_block_io(nvmestore::BLOCK_IO_READ, _io_mem, lba,
                            nr_io_blocks);

  out_value = malloc(val_len);
  assert(out_value);
  memcpy(out_value, _blk_manager->virt_addr(_io_mem), val_len);
  out_value_len = val_len;

  _meta_store->free_memory(raw_objinfo);
  return S_OK;
}

status_t persist_session::get_direct(const std ::string& key,
                                     void*               out_value,
                                     size_t&             out_value_len,
                                     buffer_t*           memory_handle)
{
  if (check_exists(key) == E_NOT_FOUND) return IKVStore::E_KEY_NOT_FOUND;
  size_t blk_sz = _blk_manager->blk_sz();

  void*  raw_objinfo = nullptr;
  size_t obj_info_length;

  status_t rc = _meta_store->get(_meta_pool, key, raw_objinfo, obj_info_length);
  if (rc != S_OK || obj_info_length < sizeof(obj_info_t)) {
    throw General_exception("get objinfo from meta_pool failed");
  }

  obj_info_t* objinfo = reinterpret_cast<obj_info_t*>(raw_objinfo);
  auto        val_len = objinfo->size;
  auto        lba     = objinfo->lba_start;

  PDBG("prepare to read lba 0x%lx with length %d", lba, val_len);
  assert(out_value);

  io_buffer_t mem;

  if (memory_handle) {  // external memory
    /* TODO: they are not nessarily equal, it memory is registered from
     * outside */
    if (out_value < memory_handle->start_vaddr()) {
      throw General_exception("out_value is not registered");
    }

    size_t offset = (size_t) out_value - (size_t)(memory_handle->start_vaddr());
    if ((val_len + offset) > memory_handle->length()) {
      throw General_exception("registered memory is not big enough");
    }

    mem = memory_handle->io_mem() + offset;
  }
  else {
    mem = reinterpret_cast<io_buffer_t>(out_value);
  }

  assert(mem);

  size_t nr_io_blocks = (val_len + blk_sz - 1) / blk_sz;

  _blk_manager->do_block_io(nvmestore::BLOCK_IO_READ, mem, lba, nr_io_blocks);

  out_value_len = val_len;

  _meta_store->free_memory(raw_objinfo);
  return S_OK;
}

status_t persist_session::put(const std::string& key,
                              const void*        value,
                              size_t             value_len,
                              unsigned int       flags)
{
  // if key exsit before overwrite it
  if (check_exists(key) == S_OK) {
    erase(key);
    return put(key, value, value_len, flags);
  }

  size_t blk_sz = _blk_manager->blk_sz();

  obj_info_t* objinfo = nullptr;  // block mapping of this obj

  may_ajust_io_mem(value_len);

  alloc_new_object(key, value_len, objinfo);

  auto lba = objinfo->lba_start;
  memcpy(_blk_manager->virt_addr(_io_mem), value,
         value_len); /* for the moment we have to memcpy */

  auto nr_io_blocks = (value_len + blk_sz - 1) / blk_sz;
  _blk_manager->do_block_io(nvmestore::BLOCK_IO_WRITE, _io_mem, lba,
                            nr_io_blocks);
  return S_OK;
}

status_t persist_session::map(std::function<int(const std::string& key,
                                                const void*        value,
                                                const size_t value_len)> f)
{
  // functor
  auto f_map = [f, this](const std::string& iter_key) -> int {
    persist_session* current_session = this;
    void*            value           = nullptr;
    size_t           value_len       = 0;

    IKVStore::lock_type_t wlock = IKVStore::STORE_LOCK_WRITE;
    // lock
    try {
      current_session->lock(iter_key, wlock, value, value_len);
    }
    catch (...) {
      throw General_exception("lock failed");
    }

    if (S_OK != f(iter_key, value, value_len)) {
      throw General_exception("apply functor failed");
    }

    // unlock
    if (S_OK != current_session->unlock((key_t) value)) {
      throw General_exception("unlock failed");
    }

    return 0;
  };
  return _meta_store->map_keys(_meta_pool, f_map);
}

status_t persist_session::map_keys(std::function<int(const std::string& key)> f)
{
  _meta_store->map_keys(_meta_pool, f);
  return S_OK;
}

persist_session::key_t persist_session::lock(const std::string& key,
                                             lock_type_t        type,
                                             void*&             out_value,
                                             size_t&            out_value_len)
{
  int operation_type = nvmestore::BLOCK_IO_NOP;

  size_t blk_sz = _blk_manager->blk_sz();

  obj_info_t* objinfo;

  if (check_exists(key) == E_NOT_FOUND) {
    if (!out_value_len) {
      throw General_exception(
          "%s: Need value length to lock a unexsiting object", __func__);
    }
    alloc_new_object(key, out_value_len, objinfo);
  }

  else {
    size_t obj_info_length;

    void*    raw_objinfo = nullptr;
    status_t rc =
        _meta_store->get(_meta_pool, key, raw_objinfo, obj_info_length);
    if (rc != S_OK || obj_info_length < sizeof(obj_info_t)) {
      throw General_exception("get objinfo from meta_pool failed");
    }
    operation_type = nvmestore::BLOCK_IO_READ;
    objinfo        = reinterpret_cast<obj_info_t*>(raw_objinfo);
  }

  auto pool = reinterpret_cast<uint64_t>(this);

  if (type == IKVStore::STORE_LOCK_READ) {
    if (!p_state_map->state_get_read_lock(pool, objinfo->block_region))
      throw General_exception("%s: unable to get read lock", __func__);
  }
  else {
    if (!p_state_map->state_get_write_lock(pool, objinfo->block_region))
      throw General_exception("%s: unable to get write lock", __func__);
  }

  auto value_len = objinfo->size;  // the length allocated before
  auto lba       = objinfo->lba_start;

  /* Prepare io mem, memory needs to be specially aligned to register in spdk*/
  size_t data_size =round_up(value_len, MB(2));
  size_t      nr_io_blocks = data_size/ blk_sz;
  assert(data_size%blk_sz == 0);

  int flags = MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB|MAP_FIXED;
  char *target_addr = ((char*) 0x900000000);
  char *data = (char *) mmap(target_addr, data_size, PROT_READ|PROT_WRITE, flags, -1, 0);

  assert(data != MAP_FAILED);
  memset(data, 0, data_size);
  int rc = spdk_mem_register(data, data_size);
  if(rc){
    PERR("register returns %d", rc);
    return 0;
  }
  io_buffer_t mem = (io_buffer_t)(data);
  out_value = data;

  /* Actual IO */
  _blk_manager->do_block_io(operation_type, mem, lba, nr_io_blocks);

  get_locked_regions().emplace(mem, key);
  PDBG("[nvmestore_session]: allocating io mem at %p, virt addr %p",
       (void*) mem, _blk_manager->virt_addr(mem));

  out_value_len = value_len;

  PDBG("NVME_store: obtained the lock for key %s", key.c_str());

  return reinterpret_cast<persist_session::key_t>(out_value);
}

status_t persist_session::unlock(persist_session::key_t key_handle)
{
  auto         pool = reinterpret_cast<uint64_t>(this);
  io_buffer_t  mem  = reinterpret_cast<io_buffer_t>(key_handle);
  std::string& key  = get_locked_regions().at(mem);

  if (check_exists(key) == E_NOT_FOUND) return IKVStore::E_KEY_NOT_FOUND;
  size_t blk_sz = _blk_manager->blk_sz();

  void*  raw_objinfo = nullptr;
  size_t obj_info_length;

  status_t rc = _meta_store->get(_meta_pool, key, raw_objinfo, obj_info_length);
  if (rc != S_OK || obj_info_length < sizeof(obj_info_t)) {
    throw General_exception("get objinfo from meta_pool failed");
  }

  obj_info_t* objinfo = reinterpret_cast<obj_info_t*>(raw_objinfo);
  auto        val_len = objinfo->size;
  auto        lba     = objinfo->lba_start;

  size_t data_size =round_up(val_len, MB(2));
  size_t      nr_io_blocks = data_size/ blk_sz;
  assert(data_size%blk_sz == 0);


  /*flush and release iomem*/
#ifdef USE_ASYNC
  uint64_t tag            = blk_dev->async_write(mem, 0, lba, nr_io_blocks);
  D_RW(objinfo)->last_tag = tag;
#else
  _blk_manager->do_block_io(nvmestore::BLOCK_IO_WRITE, mem, lba, nr_io_blocks);
#endif

  PDBG("[nvmestore_session]: freeing io mem at %p", (void*) mem);

  /* free io buffer*/
  assert(0 == spdk_mem_unregister((void *)mem, data_size));
  assert(0 == munmap((void*)mem, data_size));

  /*release the lock*/
  p_state_map->state_unlock(pool, objinfo->block_region);

  PDBG("NVME_store: released the lock for key %s\n", key.c_str());

  get_locked_regions().erase(mem);
  return S_OK;
}
}  // namespace nvmestore
