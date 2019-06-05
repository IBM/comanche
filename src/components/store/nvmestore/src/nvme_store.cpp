/*
 * (C) Copyright IBM Corporation 2017-2019. All rights reserved.
 *
 */

/*
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 * Author: Daniel Waddington
 * e-mail: daniel.waddington@ibm.com
 */
#include "nvme_store.h"

#include <libpmemobj.h>
#include <libpmemobj/base.h>
#include <libpmempool.h>
#include <functional>
#include <iostream>
#include <set>

#include <api/kvstore_itf.h>
#include <city.h>
#include <common/cycles.h>
#include <stdio.h>
#include <boost/filesystem.hpp>

#include <api/block_itf.h>
#include <api/components.h>
#include <component/base.h>
#include <core/xms.h>

#include "persist_session.h"

extern "C" {
#include "hashmap_tx.h"
}

//#define USE_ASYNC

using namespace Component;

#ifdef USE_PMEM
struct store_root_t {
  TOID(struct hashmap_tx) map; /** hashkey-> obj_info*/
  size_t pool_size;
};
// TOID_DECLARE_ROOT(struct store_root_t);
class Nvmestore_session {
 public:
  static constexpr bool option_DEBUG = false;
  using lock_type_t                  = IKVStore::lock_type_t;
  using key_t = uint64_t;  // virt_addr is used to identify each obj
  Nvmestore_session(TOID(struct store_root_t) root,
                    PMEMobjpool*              pop,
                    size_t                    pool_size,
                    std::string               path,
                    size_t                    io_mem_size,
                    nvmestore::Block_manager* blk_manager,
                    State_map*                ptr_state_map)
      : _root(root), _pop(pop), _path(path), _io_mem_size(io_mem_size),
        _blk_manager(blk_manager), p_state_map(ptr_state_map), _num_objs(0)
  {
    _io_mem = _blk_manager->allocate_io_buffer(_io_mem_size, 4096,
                                               Component::NUMA_NODE_ANY);
  }

  ~Nvmestore_session()
  {
    if (option_DEBUG) PLOG("CLOSING session");
    if (_io_mem) _blk_manager->free_io_buffer(_io_mem);
    pmemobj_close(_pop);
  }

  std::unordered_map<uint64_t, io_buffer_t>& get_locked_regions()
  {
    return _locked_regions;
  }
  std::string get_path() & { return _path; }

  /** Get persist pointer of this pool*/
  PMEMobjpool* get_pop() { return _pop; }
  /* [>* Get <]*/
  /*TOID(struct store_root_t) get_root() { return _root; }*/
  size_t get_count() { return _num_objs; }

  void alloc_new_object(const std::string& key,
                        size_t             value_len,
                        TOID(struct obj_info) & out_blkmeta);

  /** Erase Objects*/
  status_t erase(const std::string& key);

  /** Put and object*/
  status_t put(const std::string& key,
               const void*        valude,
               size_t             value_len,
               unsigned int       flags);

  /** Get an object*/
  status_t get(const std::string& key, void*& out_value, size_t& out_value_len);

  status_t get_direct(const std ::string& key,
                      void*               out_value,
                      size_t&             out_value_len,
                      buffer_t*           memory_handle);

  key_t    lock(const std::string& key,
                lock_type_t        type,
                void*&             out_value,
                size_t&            out_value_len);
  status_t unlock(key_t obj_key);

  status_t map(std::function<int(const std::string& key,
                                 const void*        value,
                                 const size_t       value_len)> f);
  status_t map_keys(std::function<int(const std::string& key)> f);

 private:
  // for meta_pmem only
  TOID(struct store_root_t) _root;
  PMEMobjpool* _pop;  // the pool for mapping

  size_t                    _pool_size;
  std::string               _path;
  uint64_t                  _io_mem;      /** dynamic iomem for put/get */
  size_t                    _io_mem_size; /** io memory size */
  nvmestore::Block_manager* _blk_manager;
  State_map*                p_state_map;

  /** Session locked, io_buffer_t(virt_addr) -> pool hashkey of obj*/
  std::unordered_map<io_buffer_t, uint64_t> _locked_regions;
  size_t                                    _num_objs;

  status_t may_ajust_io_mem(size_t value_len);
};

/**
 * Create a entry in the pool and allocate space
 *
 * @param session
 * @param value_len
 * @param out_blkmeta [out] block mapping info of this obj
 *
 * TODO This allocate memory using regions */
void Nvmestore_session::alloc_new_object(const std::string& key,
                                         size_t             value_len,
                                         TOID(struct obj_info) & out_blkmeta)
{
  auto& root        = this->_root;
  auto& pop         = this->_pop;
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

status_t Nvmestore_session::erase(const std::string& key)
{
  uint64_t hashkey = CityHash64(key.c_str(), key.length());
  auto&    root    = this->_root;
  auto&    pop     = this->_pop;

  uint64_t pool = reinterpret_cast<uint64_t>(this);

  TOID(struct obj_info) objinfo;
  try {
    objinfo = hm_tx_get(pop, D_RW(root)->map, hashkey);
    if (OID_IS_NULL(objinfo.oid)) return NVME_store::E_KEY_NOT_FOUND;

    /* get hold of write lock to remove */
    if (!p_state_map->state_get_write_lock(pool, D_RO(objinfo)->handle))
      throw API_exception("unable to remove, value locked");

    PDBG("Tring to Remove obj with obj %p,handle %p", D_RO(objinfo),
         D_RO(objinfo)->handle);

    // Free objinfo
    objinfo = hm_tx_remove(pop, D_RW(root)->map,
                           hashkey); /* could be optimized to not re-lookup */
    TX_BEGIN(_pop) { TX_FREE(objinfo); }
    TX_ONABORT
    {
      throw General_exception("TX abort (%s) when free objinfo record",
                              pmemobj_errormsg());
    }
    TX_END

    if (OID_IS_NULL(objinfo.oid))
      throw API_exception("hm_tx_remove with key(%lu) failed unexpectedly %s",
                          hashkey, pmemobj_errormsg());

    // Free block range in the blk_alloc
    _blk_manager->free_blk_region(D_RO(objinfo)->lba_start,
                                  D_RO(objinfo)->handle);

    p_state_map->state_remove(pool, D_RO(objinfo)->handle);
    _num_objs -= 1;
  }
  catch (...) {
    throw General_exception("hm_tx_remove failed unexpectedly");
  }
  return S_OK;
}

status_t Nvmestore_session::may_ajust_io_mem(size_t value_len)
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

status_t Nvmestore_session::put(const std::string& key,
                                const void*        value,
                                size_t             value_len,
                                unsigned int       flags)
{
  uint64_t hashkey = CityHash64(key.c_str(), key.length());
  size_t   blk_sz  = _blk_manager->blk_sz();

  auto root = _root;
  auto pop  = _pop;

  TOID(struct obj_info) blkmeta;  // block mapping of this obj

  if (hm_tx_lookup(pop, D_RO(root)->map, hashkey)) {
    PLOG("overriting exsiting obj");
    erase(key);
    return put(key, value, value_len, flags);
  }

  may_ajust_io_mem(value_len);

  alloc_new_object(key, value_len, blkmeta);

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

status_t Nvmestore_session::get(const std::string& key,
                                void*&             out_value,
                                size_t&            out_value_len)
{
  uint64_t hashkey = CityHash64(key.c_str(), key.length());
  size_t   blk_sz  = _blk_manager->blk_sz();

  auto  root = _root;
  auto& pop  = _pop;

  TOID(struct obj_info) objinfo;
  // TODO: can write to a shadowed copy
  try {
    objinfo = hm_tx_get(pop, D_RW(root)->map, hashkey);
    if (OID_IS_NULL(objinfo.oid)) return IKVStore::E_KEY_NOT_FOUND;

    auto val_len = D_RO(objinfo)->size;
    auto lba     = D_RO(objinfo)->lba_start;

#ifdef USE_ASYNC
    uint64_t tag = D_RO(objinfo)->last_tag;
    while (!blk_dev->check_completion(tag))
      cpu_relax(); /* check the last completion, TODO: check each time makes the
                      get slightly slow () */
#endif
    PDBG("prepare to read lba %d with length %d, key %lx", lba, val_len,
         hashkey);

    may_ajust_io_mem(val_len);
    size_t nr_io_blocks = (val_len + blk_sz - 1) / blk_sz;

    _blk_manager->do_block_io(nvmestore::BLOCK_IO_READ, _io_mem, lba,
                              nr_io_blocks);

    out_value = malloc(val_len);
    assert(out_value);
    memcpy(out_value, _blk_manager->virt_addr(_io_mem), val_len);
    out_value_len = val_len;
  }
  catch (...) {
    throw General_exception("hm_tx_get failed unexpectedly");
  }
  return S_OK;
}

status_t Nvmestore_session::get_direct(const std::string& key,
                                       void*              out_value,
                                       size_t&            out_value_len,
                                       buffer_t*          memory_handle)
{
  uint64_t hashkey = CityHash64(key.c_str(), key.length());
  size_t   blk_sz  = _blk_manager->blk_sz();

  auto root = _root;
  auto pop  = _pop;

  TOID(struct obj_info) objinfo;
  try {
    cpu_time_t start = rdtsc();
    objinfo          = hm_tx_get(pop, D_RW(root)->map, hashkey);
    if (OID_IS_NULL(objinfo.oid)) return IKVStore::E_KEY_NOT_FOUND;

    auto val_len = static_cast<size_t>(D_RO(objinfo)->size);
    auto lba     = static_cast<lba_t>(D_RO(objinfo)->lba_start);

    cpu_time_t cycles_for_hm = rdtsc() - start;

    PLOG("checked hxmap read latency took %ld cycles (%f usec) per hm access",
         cycles_for_hm, cycles_for_hm / 2400.0f);

#ifdef USE_ASYNC
    uint64_t tag = D_RO(objinfo)->last_tag;
    while (!blk_dev->check_completion(tag))
      cpu_relax(); /* check the last completion, TODO: check each time makes the
                      get slightly slow () */
#endif

    PDBG("prepare to read lba %lu with length %lu", lba, val_len);
    assert(out_value);

    io_buffer_t mem;

    if (memory_handle) {  // external memory
      /* TODO: they are not nessarily equal, it memory is registered from
       * outside */
      if (out_value < memory_handle->start_vaddr()) {
        throw General_exception("out_value is not registered");
      }

      size_t offset =
          (size_t) out_value - (size_t)(memory_handle->start_vaddr());
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

    start = rdtsc();

    _blk_manager->do_block_io(nvmestore::BLOCK_IO_READ, mem, lba, nr_io_blocks);

    cpu_time_t cycles_for_iop = rdtsc() - start;
    PDBG("prepare to read lba %lu with nr_blocks %lu", lba, nr_io_blocks);
    PDBG("checked read latency took %ld cycles (%f usec) per IOP",
         cycles_for_iop, cycles_for_iop / 2400.0f);
    out_value_len = val_len;
  }
  catch (...) {
    throw General_exception("hm_tx_get failed unexpectedly");
  }
  return S_OK;
}

Nvmestore_session::key_t Nvmestore_session::lock(const std::string& key,
                                                 lock_type_t        type,
                                                 void*&             out_value,
                                                 size_t& out_value_len)
{
  uint64_t hashkey        = CityHash64(key.c_str(), key.length());
  auto&    root           = _root;
  auto&    pop            = _pop;
  int      operation_type = nvmestore::BLOCK_IO_NOP;

  size_t blk_sz = _blk_manager->blk_sz();
  auto   pool   = reinterpret_cast<uint64_t>(this);

  TOID(struct obj_info) objinfo;
  try {
    objinfo = hm_tx_get(pop, D_RW(root)->map, hashkey);

    if (!OID_IS_NULL(objinfo.oid)) {
#ifdef USE_ASYNC
      /* there might be pending async write for this object */
      uint64_t tag = D_RO(objinfo)->last_tag;
      while (!blk_dev->check_completion(tag))
        cpu_relax(); /* check the last completion */
#endif
      operation_type = nvmestore::BLOCK_IO_READ;
    }
    else {
      if (!out_value_len) {
        throw General_exception(
            "%s: Need value length to lock a unexsiting object", __func__);
      }
      alloc_new_object(key, out_value_len, objinfo);
    }

    if (type == IKVStore::STORE_LOCK_READ) {
      if (!p_state_map->state_get_read_lock(pool, D_RO(objinfo)->handle))
        throw General_exception("%s: unable to get read lock", __func__);
    }
    else {
      if (!p_state_map->state_get_write_lock(pool, D_RO(objinfo)->handle))
        throw General_exception("%s: unable to get write lock", __func__);
    }

    auto handle    = D_RO(objinfo)->handle;
    auto value_len = D_RO(objinfo)->size;  // the length allocated before
    auto lba       = D_RO(objinfo)->lba_start;

    /* fetch the data to block io mem */
    size_t      nr_io_blocks = (value_len + blk_sz - 1) / blk_sz;
    io_buffer_t mem          = _blk_manager->allocate_io_buffer(
        nr_io_blocks * blk_sz, 4096, Component::NUMA_NODE_ANY);

    _blk_manager->do_block_io(operation_type, mem, lba, nr_io_blocks);

    get_locked_regions().emplace(mem, hashkey);
    PDBG("[nvmestore_session]: allocating io mem at %p, virt addr %p",
         (void*) mem, _blk_manager->virt_addr(mem));

    /* set output values */
    out_value     = _blk_manager->virt_addr(mem);
    out_value_len = value_len;
  }
  catch (...) {
    PERR("NVME_store: lock failed");
  }

  PDBG("NVME_store: obtained the lock");

  return reinterpret_cast<Nvmestore_session::key_t>(out_value);
}

status_t Nvmestore_session::unlock(Nvmestore_session::key_t key_handle)
{
  auto        root    = _root;
  auto        pop     = _pop;
  auto        pool    = reinterpret_cast<uint64_t>(this);
  io_buffer_t mem     = reinterpret_cast<io_buffer_t>(key_handle);
  uint64_t    hashkey = get_locked_regions().at(mem);

  size_t blk_sz = _blk_manager->blk_sz();

  TOID(struct obj_info) objinfo;
  try {
    objinfo = hm_tx_get(pop, D_RW(root)->map, (uint64_t) hashkey);
    if (OID_IS_NULL(objinfo.oid)) return IKVStore::E_KEY_NOT_FOUND;

    auto val_len = D_RO(objinfo)->size;
    auto lba     = D_RO(objinfo)->lba_start;

    size_t nr_io_blocks = (val_len + blk_sz - 1) / blk_sz;

    /*flush and release iomem*/
#ifdef USE_ASYNC
    uint64_t tag            = blk_dev->async_write(mem, 0, lba, nr_io_blocks);
    D_RW(objinfo)->last_tag = tag;
#else
    _blk_manager->do_block_io(nvmestore::BLOCK_IO_WRITE, mem, lba,
                              nr_io_blocks);
#endif

    PDBG("[nvmestore_session]: freeing io mem at %p", (void*) mem);
    _blk_manager->free_io_buffer(mem);

    /*release the lock*/
    p_state_map->state_unlock(pool, D_RO(objinfo)->handle);

    PDBG("NVME_store: released the lock");
  }
  catch (...) {
    throw General_exception("hm_tx_get failed unexpectedly or iomem not found");
  }
  return S_OK;
}

status_t Nvmestore_session::map(std::function<int(const std::string& key,
                                                  const void*        value,
                                                  const size_t value_len)> f)
{
  size_t blk_sz = _blk_manager->blk_sz();

  auto& root = _root;
  auto& pop  = _pop;

  TOID(struct obj_info) objinfo;

  // functor
  auto f_map = [f, pop, root](uint64_t hashkey, void* arg) -> int {
    Nvmestore_session* session   = reinterpret_cast<Nvmestore_session*>(arg);
    void*              value     = nullptr;
    size_t             value_len = 0;

    TOID(struct obj_info) objinfo;
    objinfo             = hm_tx_get(pop, D_RW(root)->map, (uint64_t) hashkey);
    const char* key_str = D_RO(D_RO(objinfo)->key_data);
    std::string key(key_str);

    IKVStore::lock_type_t wlock = IKVStore::STORE_LOCK_WRITE;
    // lock
    try {
      session->lock(key, wlock, value, value_len);
    }
    catch (...) {
      throw General_exception("lock failed");
    }

    if (S_OK != f(key, value, value_len)) {
      throw General_exception("apply functor failed");
    }

    // unlock
    if (S_OK != session->unlock((key_t) value)) {
      throw General_exception("unlock failed");
    }

    return 0;
  };

  TX_BEGIN(pop)
  {
    // lock/apply/ and unlock
    hm_tx_foreachkey(pop, D_RW(root)->map, f_map, this);
  }
  TX_ONABORT { throw General_exception("Map for each failed"); }
  TX_END

  return S_OK;
}

status_t Nvmestore_session::map_keys(
    std::function<int(const std::string& key)> f)
{
  auto& root = _root;
  auto& pop  = _pop;

  TOID(struct obj_info) objinfo;

  // functor
  auto f_map = [f, pop, root](uint64_t hashkey, void* arg) -> int {
    TOID(struct obj_info) objinfo;
    objinfo             = hm_tx_get(pop, D_RW(root)->map, (uint64_t) hashkey);
    const char* key_str = D_RO(D_RO(objinfo)->key_data);
    std::string key(key_str);

    if (S_OK != f(key)) {
      throw General_exception("apply functor failed");
    }

    return 0;
  };

  TX_BEGIN(pop) { hm_tx_foreachkey(pop, D_RW(root)->map, f_map, this); }
  TX_ONABORT { throw General_exception("MapKeys for each failed"); }
  TX_END

  return S_OK;
}
#endif

// using open_session_t = Nvmestore_session;
using open_session_t = nvmestore::persist_session;

struct tls_cache_t {
  open_session_t* session;
};

static __thread tls_cache_t tls_cache = {nullptr};
std::set<open_session_t*>   g_sessions;

static open_session_t* get_session(
    IKVStore::pool_t pid)  // open_session_t * session)
{
  open_session_t* session = reinterpret_cast<open_session_t*>(pid);
  if (session == tls_cache.session && session != nullptr) return session;

  if (g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store:: invalid pool identifier");

  return session;
}

static int check_pool(const char* path)
{
  PLOG("check_pool: %s", path);
  PMEMpoolcheck*                ppc;
  struct pmempool_check_status* status;

  struct pmempool_check_args args;
  args.path        = path;
  args.backup_path = NULL;
  args.pool_type   = PMEMPOOL_POOL_TYPE_DETECT;
  args.flags       = PMEMPOOL_CHECK_FORMAT_STR | PMEMPOOL_CHECK_REPAIR |
               PMEMPOOL_CHECK_VERBOSE;

  if ((ppc = pmempool_check_init(&args, sizeof(args))) == NULL) {
    perror("pmempool_check_init");
    return -1;
  }

  /* perform check and repair, answer 'yes' for each question */
  while ((status = pmempool_check(ppc)) != NULL) {
    switch (status->type) {
      case PMEMPOOL_CHECK_MSG_TYPE_ERROR:
        printf("%s\n", status->str.msg);
        break;
      case PMEMPOOL_CHECK_MSG_TYPE_INFO:
        printf("%s\n", status->str.msg);
        break;
      case PMEMPOOL_CHECK_MSG_TYPE_QUESTION:
        printf("%s\n", status->str.msg);
        status->str.answer = "yes";
        break;
      default:
        pmempool_check_end(ppc);
        throw General_exception("pmempool_check failed");
    }
  }

  /* finalize the check and get the result */
  int ret = pmempool_check_end(ppc);
  switch (ret) {
    case PMEMPOOL_CHECK_RESULT_CONSISTENT:
    case PMEMPOOL_CHECK_RESULT_REPAIRED:
      PLOG("pool (%s) checked OK!", path);
      return 0;
  }

  return 1;
}

NVME_store::NVME_store(const std::string& owner,
                       const std::string& name,
                       const std::string& pci,
                       const std::string& pm_path,
                       const std::string& config_persist_type)
    : _pm_path(pm_path), _blk_manager(pci, pm_path)
{
#ifdef USE_PMEM
  if (config_persist_type == "pmem") {
    _meta_persist_type = PERSIST_PMEM;
    PINF("NVMe_store, using persist type %s", config_persist_type.c_str());

    PLOG("PMEMOBJ_MAX_ALLOC_SIZE: %lu MB", REDUCE_MB(PMEMOBJ_MAX_ALLOC_SIZE));
  }
#endif
  IBase* comp;
  if (config_persist_type == "filestore") {
    _meta_persist_type = PERSIST_FILE;
    comp = load_component("libcomanche-storefile.so", filestore_factory);
  }
  else if (config_persist_type == "hstore") {
    _meta_persist_type = PERSIST_HSTORE;
    comp = load_component("libcomanche-hstore.so", hstore_factory);
    throw API_exception("not implemented");
  }
  else {
    throw API_exception("Option %s not supported", config_persist_type.c_str());
  }
  if (!comp)
    throw General_exception("unable to initialize Dawn backend component");

  IKVStore_factory* fact =
      (IKVStore_factory*) comp->query_interface(IKVStore_factory::iid());
  assert(fact);

  unsigned debug_level = 0;
  if (_meta_persist_type ==
      PERSIST_FILE) { /* components that support debug level */
    std::map<std::string, std::string> params;
    params["pm_path"] = pm_path + "meta/";
    _meta_store       = fact->create(debug_level, params);
  }
  else if (_meta_persist_type == PERSIST_HSTORE) {
    // TODO refer to dawn
    throw API_exception("hstore init not implemented");
  }
  else {
    _meta_store = fact->create("owner", "name");
  }
  fact->release_ref();

  // path
  if (_pm_path.back() != '/') _pm_path += "/";
}

NVME_store::~NVME_store()
{
  if (option_DEBUG) PLOG("deleting NVME store");
  _meta_store->release_ref();
}

IKVStore::pool_t NVME_store::create_pool(const std::string& name,
                                         const size_t       size,
                                         unsigned int       flags,
                                         uint64_t           args)
{
  int ret = 0;

  // TODO: need to check size
  const std::string& fullpath = _pm_path + name;

  PINF("[NVME_store]::create_pool fullpath=%s name=%s", fullpath.c_str(),
       name.c_str());

#if USE_PMEM
  if (_meta_persist_type == PERSIST_PMEM) {  // deprecated

    size_t max_sz_hxmap = MB(500);  // this can fit 1M objects (obj_info_t)
    /* open existing pool */
    PMEMobjpool* pop = nullptr;  // pool to allocate all mapping
    pop = pmemobj_open(fullpath.c_str(), POBJ_LAYOUT_NAME(nvme_store));

    if (!pop) {
      PLOG("creating new pool: %s", name.c_str());

      boost::filesystem::path p(fullpath);
      boost::filesystem::create_directories(p.parent_path());

      pop = pmemobj_create(fullpath.c_str(), POBJ_LAYOUT_NAME(nvme_store),
                           max_sz_hxmap, 0666);
    }

    if (not pop) return POOL_ERROR;

    /* see:
     * https://github.com/pmem/pmdk/blob/stable-1.4/src/examples/libpmemobj/map/kv_server.c
     */
    assert(pop);
    TOID(struct store_root_t) root = POBJ_ROOT(pop, struct store_root_t);
    assert(!TOID_IS_NULL(root));

    if (D_RO(root)->map.oid.off == 0) {
      /* create hash table if it does not exist */
      TX_BEGIN(pop)
      {
        if (hm_tx_create(pop, &D_RW(root)->map, nullptr))
          throw General_exception("hm_tx_create failed unexpectedly");
        D_RW(root)->pool_size = size;
      }
      TX_ONABORT { ret = -1; }
      TX_END
    }
    // Check
    if (hm_tx_check(pop, D_RO(root)->map))
      throw General_exception("hm_tx_check failed unexpectedly");

    open_session_t* session = new open_session_t(
        root, pop, size, fullpath, DEFAULT_IO_MEM_SIZE, &_blk_manager, &_sm);
  }
#endif
  size_t estimated_obj_map_size = MB(4);  // 32B per entry, that's 2^17
  IKVStore::pool_t obj_info_pool =
      _meta_store->create_pool(name, estimated_obj_map_size, flags);
  if (obj_info_pool == POOL_ERROR) {
    throw General_exception("Creating objmap pool failed");
  }
  open_session_t* session =
      new open_session_t(_meta_store, obj_info_pool, fullpath,
                         DEFAULT_IO_MEM_SIZE, &_blk_manager, &_sm);

  g_sessions.insert(session);

  return reinterpret_cast<uint64_t>(session);
}

IKVStore::pool_t NVME_store::open_pool(const std::string& name,
                                       unsigned int       flags)
{
  const std::string& fullpath = _pm_path + name;

#ifdef USE_PMEM
  PMEMobjpool* pop;                     // pool to allocate all mapping
  size_t       max_sz_hxmap = MB(500);  // this can fit 1M objects (obj_info_t)

  PINF("NVME_store::open_pool name=%s", name.c_str());

  /* if trying to open a unclosed pool!*/
  for (auto iter : g_sessions) {
    if (iter->get_path() == fullpath) {
      PWRN("nvmestore: try to reopen a pool!");
      return reinterpret_cast<uint64_t>(iter);
    }
  }

  if (access(fullpath.c_str(), F_OK) != 0) {
    throw General_exception("nvmestore: pool not existing at path %s",
                            fullpath.c_str());
  }
  else {
    PLOG("Opening existing Pool: %s", name.c_str());

    if (check_pool(fullpath.c_str()) != 0)
      throw General_exception("pool check failed");

    pop = pmemobj_open(fullpath.c_str(), POBJ_LAYOUT_NAME(nvme_store));
    if (not pop)
      throw General_exception("failed to re-open pool - %s\n",
                              pmemobj_errormsg());
  }

  TOID(struct store_root_t) root = POBJ_ROOT(pop, struct store_root_t);
  assert(!TOID_IS_NULL(root));

  assert(D_RO(root)->map.oid.off != 0);
  /*TODO; in caffe workload the poolsize is not persist somehow*/
  if (D_RO(root)->pool_size == 0) {
    PWRN("nvmestore: pool size is ZERO!");
  }
  PLOG("Using existing root, pool size =  %lu:", D_RO(root)->pool_size);
  if (hm_tx_init(pop, D_RW(root)->map))
    throw General_exception("hm_tx_init failed unexpectedly");
  size_t          pool_size = D_RO(root)->pool_size;
  open_session_t* session   = new open_session_t(
      root, pop, pool_size, fullpath, DEFAULT_IO_MEM_SIZE, &_blk_manager, &_sm);

#else
  IKVStore::pool_t obj_info_pool = _meta_store->open_pool(name, flags);
  if (obj_info_pool == POOL_ERROR) {
    throw General_exception("objmap pool: failed during opening ");
  }

  open_session_t* session =
      new open_session_t(_meta_store, obj_info_pool, fullpath,
                         DEFAULT_IO_MEM_SIZE, &_blk_manager, &_sm);

#endif

  g_sessions.insert(session);

  return reinterpret_cast<uint64_t>(session);
}

status_t NVME_store::close_pool(pool_t pid)
{
  open_session_t* session = reinterpret_cast<open_session_t*>(pid);

  if (g_sessions.find(session) == g_sessions.end()) {
    return E_INVAL;
  }

#ifdef USE_PMEM
  auto& pop = session->get_pop();
  pmemobj_close(pop);
#endif
  if (S_OK != _meta_store->close_pool(session->get_obj_info_pool())) {
    throw General_exception("Close objmap pool failed");
  }

  delete session;

  // TODO: correct?
  g_sessions.erase(session);
  PLOG("NVME_store::closed pool (%lx)", pid);
  return S_OK;
}

status_t NVME_store::delete_pool(const std::string& name)
{
  // return S_OK on success, E_POOL_NOT_FOUND, E_ALREADY_OPEN
  const std::string& fullpath = _pm_path + name;

  /* if trying to open a unclosed pool!*/
  for (auto iter : g_sessions) {
    if (iter->get_path() == fullpath) {
      PWRN("nvmestore: try to delete an opened pool!");
      return E_ALREADY_OPEN;
    }
  }
#ifdef USE_PMEM
  if (access(fullpath.c_str(), F_OK) != 0) {
    PWRN("nvmestore: pool doesn't exsit!");
    return E_POOL_NOT_FOUND;
  }

  // TODO should clean the blk_allocator and blk dev (reference) here?
  //_blk_alloc->resize(0, 0);

  if (pmempool_rm(fullpath.c_str(), 0))
    throw General_exception("unable to delete pool (%s)", fullpath.c_str());
  PLOG("pool deleted: %s", fullpath.c_str());
#endif

  // TODO erase from main store
  if (S_OK != _meta_store->delete_pool(name)) {
    throw General_exception("objmap pool-  failed when deleting");
  }
  return S_OK;
}

/*
 * when using NVMe, only insert the block range descriptor into the mapping
 */
status_t NVME_store::put(IKVStore::pool_t   pool,
                         const std::string& key,
                         const void*        value,
                         size_t             value_len,
                         unsigned int       flags)
{
  open_session_t* session = reinterpret_cast<open_session_t*>(pool);

  if (g_sessions.find(session) == g_sessions.end())
    // TODO: resize the allocation
    throw API_exception("NVME_store::put invalid pool identifier");

  return session->put(key, value, value_len, flags);
}

status_t NVME_store::get(const pool_t       pool,
                         const std::string& key,
                         void*&             out_value,
                         size_t&            out_value_len)
{
  open_session_t* session = reinterpret_cast<open_session_t*>(pool);

  if (g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store::put invalid pool identifier");

  return session->get(key, out_value, out_value_len);
}

status_t NVME_store::get_direct(const pool_t       pool,
                                const std::string& key,
                                void*              out_value,
                                size_t&            out_value_len,
                                Component::IKVStore::memory_handle_t handle)
{
  open_session_t* session = reinterpret_cast<open_session_t*>(pool);

  if (g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store::put invalid pool identifier");

  return session->get_direct(key, out_value, out_value_len,
                             reinterpret_cast<buffer_t*>(handle));
}

static_assert(sizeof(IKVStore::memory_handle_t) == sizeof(io_buffer_t),
              "cast may not work");

/*
 * Direct memory for NVMeStore.
 * Only used for the case when memory is pinned/aligned but not from spdk, e.g.
 * cudadma should be 2MB aligned in both phsycial and virtual
 * */
IKVStore::memory_handle_t NVME_store::allocate_direct_memory(void*& vaddr,
                                                             size_t len)
{
  io_buffer_t io_mem;
  size_t      blk_sz = _blk_manager.blk_sz();
  io_mem = _blk_manager.allocate_io_buffer(round_up(len, blk_sz), 4096,
                                           Component::NUMA_NODE_ANY);
  if (io_mem == 0)
    throw API_exception("NVME_store:: direct memory allocation failed");
  vaddr = _blk_manager.virt_addr(io_mem);

  buffer_t* buffer = new buffer_t(len, io_mem, vaddr);

  auto handle = reinterpret_cast<IKVStore::memory_handle_t>(buffer);
  /* save this this registration */

  return handle;
}

status_t NVME_store::free_direct_memory(memory_handle_t handle)
{
  buffer_t* buffer = reinterpret_cast<buffer_t*>(handle);
  _blk_manager.free_io_buffer(buffer->io_mem());
  delete buffer;
  return S_OK;
}

IKVStore::memory_handle_t NVME_store::register_direct_memory(void*  vaddr,
                                                             size_t len)
{
  memset(vaddr, 0, len);

  addr_t      phys_addr = xms_get_phys(vaddr);
  io_buffer_t io_mem =
      _blk_manager.register_memory_for_io(vaddr, phys_addr, len);
  buffer_t* buffer = new buffer_t(len, io_mem, vaddr);

  auto handle = reinterpret_cast<IKVStore::memory_handle_t>(buffer);
  /* save this this registration */
  if (io_mem)
    PINF("Register vaddr %p with paddr %lu, handle %lu", vaddr, phys_addr,
         io_mem);
  else
    PERR("%s: register user allocated memory failed", __func__);

  return handle;
}

status_t NVME_store::unregister_direct_memory(memory_handle_t handle)
{
  buffer_t* buffer = reinterpret_cast<buffer_t*>(handle);
  _blk_manager.unregister_memory_for_io(buffer->start_vaddr(),
                                        buffer->length());
  delete buffer;
  return S_OK;
}

/*
 * For nvmestore, data is not necessarily in main memory.
 * Lock will allocate iomem and load data from nvme first.
 * Unlock will will free it
 */
IKVStore::key_t NVME_store::lock(const pool_t       pool,
                                 const std::string& key,
                                 lock_type_t        type,
                                 void*&             out_value,
                                 size_t&            out_value_len)
{
  open_session_t* session = get_session(pool);

  session->lock(key, type, out_value, out_value_len);
  PDBG("[nvmestore_lock] %p", out_value);
  return reinterpret_cast<Component::IKVStore::key_t>(out_value);
}

/*
 * For nvmestore, data is not necessarily in main memory.
 * Lock will allocate iomem and load data from nvme first.
 * Unlock will will free it
 */
status_t NVME_store::unlock(const pool_t pool, key_t key_handle)
{
  open_session_t* session = get_session(pool);

  session->unlock((uint64_t) key_handle);  // i.e. virt addr
  return S_OK;
}

size_t NVME_store::count(const pool_t pool)
{
  open_session_t* session = get_session(pool);
  return session->get_count();
}

status_t NVME_store::erase(const pool_t pool, const std::string& key)
{
  open_session_t* session = get_session(pool);

  return session->erase(key);
}

status_t NVME_store::map(const pool_t                               pool,
                         std::function<int(const std::string& key,
                                           const void*        value,
                                           const size_t value_len)> function)
{
  open_session_t* session = get_session(pool);

  return session->map(function);
}

status_t NVME_store::map_keys(
    const pool_t                               pool,
    std::function<int(const std::string& key)> function)
{
  open_session_t* session = get_session(pool);

  return session->map_keys(function);
}

void NVME_store::debug(const pool_t pool, unsigned cmd, uint64_t arg) {}

/**
 * Factory entry point
 *
 */
extern "C" void* factory_createInstance(Component::uuid_t& component_id)
{
  if (component_id == NVME_store_factory::component_id()) {
    return reinterpret_cast<void*>(new NVME_store_factory());
  }
  else
    return NULL;
}

void* NVME_store_factory::query_interface(Component::uuid_t& itf_uuid)
{
  if (itf_uuid == Component::IKVStore_factory::iid()) {
    return this;
  }
  else
    return NULL;  // we don't support this interface
}

void NVME_store_factory::unload() { delete this; }

IKVStore* NVME_store_factory::create(unsigned debug_level,
                                     std::map<std::string, std::string>& params)
{
  auto& pci = params["pci"];

  if (pci.size() != 7 || pci[2] != ':' || pci[5] != '.') {
    throw Constructor_exception(
        "Parameter '%s' does not look like a PCI address", pci.c_str());
  }

  std::string meta_persist_type = params.find("persist_type") == params.end()
                                      ? "filestore"
                                      : params["persist_type"];
  Component::IKVStore* obj = static_cast<Component::IKVStore*>(
      new NVME_store(params["owner"], params["name"], params["pci"],
                     params["pm_path"], meta_persist_type));
  obj->add_ref();
  return obj;
}
