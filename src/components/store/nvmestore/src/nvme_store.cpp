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

#include "state_map.h"

extern "C" {
#include "hashmap_tx.h"
}

//#define USE_ASYNC

using namespace Component;

struct store_root_t {
  TOID(struct hashmap_tx) map;  // name mappings
  size_t pool_size;
};
// TOID_DECLARE_ROOT(struct store_root_t);

class Nvmestore_session {
 public:
   static constexpr bool option_DEBUG = false;
  using lock_type_t = IKVStore::lock_type_t;
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
    if(option_DEBUG) PLOG("CLOSING session");
    if (_io_mem) _blk_manager->free_io_buffer(_io_mem);
    pmemobj_close(_pop);
  }
  void alloc_new_object(uint64_t hashkey,
                        size_t   value_len,
                        TOID(struct block_range) & out_blkmeta);

  std::unordered_map<uint64_t, io_buffer_t>& get_locked_regions()
  {
    return _locked_regions;
  }
  std::string get_path() & { return _path; }

  /** Get persist pointer of this pool*/
  PMEMobjpool* get_pop() { return _pop; }
  /** Get */
  TOID(struct store_root_t) get_root() { return _root; }

  /** Erase Objects*/
  status_t erase(uint64_t hashkey);

  /** Put and object*/
  status_t put(uint64_t     hashkey,
               const void*  valude,
               size_t       value_len,
               unsigned int flags);

  /** Get an object*/
  status_t get(uint64_t hashkey, void*& out_value, size_t& out_value_len);

  status_t get_direct(uint64_t hashkey, void* out_value, size_t& out_value_len, buffer_t * memory_handle);

  status_t lock(uint64_t    hashkey,
                lock_type_t type,
                void*&      out_value,
                size_t&     out_value_len);
  status_t unlock(uint64_t hashkey);

 private:
  TOID(struct store_root_t) _root;
  PMEMobjpool* _pop;  // the pool for mapping
  size_t       _pool_size;
  std::string  _path;
  uint64_t
      _io_mem;  // io memory for lock/unload TODO: this should be thead-safe,
                // this should be large enough store the allocated value
  size_t _io_mem_size;  // io memory size which will an object;
  nvmestore::Block_manager* _blk_manager;
  State_map*                p_state_map;

  /** Session locked, io_buffer_t(virt_addr) -> pool hashkey of obj*/
  std::unordered_map<io_buffer_t, uint64_t> _locked_regions;
  size_t                                    _num_objs;

  status_t may_ajust_io_mem(size_t value_len);
};

status_t Nvmestore_session::erase(uint64_t hashkey)
{
  auto& root = this->_root;
  auto& pop  = this->_pop;

  uint64_t pool = reinterpret_cast<uint64_t>(this);

  TOID(struct block_range) blk_info;
  try {
    blk_info = hm_tx_get(pop, D_RW(root)->map, hashkey);
    if (OID_IS_NULL(blk_info.oid)) return NVME_store::E_KEY_NOT_FOUND;

    /* get hold of write lock to remove */
    if (!p_state_map->state_get_write_lock(pool, D_RO(blk_info)->handle))
      throw API_exception("unable to remove, value locked");

    _blk_manager->free_blk_region(D_RO(blk_info)->lba_start,
                                  D_RO(blk_info)->handle);

    blk_info = hm_tx_remove(pop, D_RW(root)->map,
                            hashkey); /* could be optimized to not re-lookup */
    if (OID_IS_NULL(blk_info.oid))
      throw API_exception("hm_tx_remove failed unexpectedly");
    p_state_map->state_remove(pool, D_RO(blk_info)->handle);
    _num_objs -= 1;
  }
  catch (...) {
    throw General_exception("hm_tx_remove failed unexpectedly");
  }
  return S_OK;
};

status_t Nvmestore_session::may_ajust_io_mem(size_t value_len){
  /*  increase IO buffer sizes when value size is large*/
  // TODO: need lock
  if (value_len > _io_mem_size) {
    size_t new_io_mem_size = _io_mem_size;

    while(new_io_mem_size < value_len){
      new_io_mem_size*=2;
    }

    _io_mem_size = new_io_mem_size;
    _blk_manager->free_io_buffer(_io_mem);

    _io_mem         = _blk_manager->allocate_io_buffer(
        _io_mem_size, 4096, Component::NUMA_NODE_ANY);
    if(option_DEBUG) PINF("[Nvmestore_session]: incresing IO mem size %lu at %lx", new_io_mem_size, _io_mem);
  }
  return S_OK;
}

status_t Nvmestore_session::put(uint64_t     hashkey,
                                const void*  value,
                                size_t       value_len,
                                unsigned int flags)
{
  size_t blk_sz = _blk_manager->blk_sz();

  auto root = _root;
  auto pop  = _pop;

  TOID(struct block_range) blkmeta;  // block mapping of this obj

  may_ajust_io_mem(value_len);

  if (hm_tx_lookup(pop, D_RO(root)->map, hashkey))
    return IKVStore::E_KEY_EXISTS;

  alloc_new_object(hashkey, value_len, blkmeta);

  memcpy(_blk_manager->virt_addr(_io_mem), value,
         value_len); /* for the moment we have to memcpy */

#ifdef USE_ASYNC
#error("use_sync is deprecated")
  // TODO: can the free be triggered by callback?
  uint64_t tag = blk_dev->async_write(session->io_mem, 0, lba, nr_io_blocks);
  D_RW(blk_info)->last_tag = tag;
#else
  auto nr_io_blocks = (value_len + blk_sz - 1) / blk_sz;
  _blk_manager->do_block_io(nvmestore::BLOCK_IO_WRITE, _io_mem,
                            D_RO(blkmeta)->lba_start, nr_io_blocks);
#endif
  _num_objs += 1;
  return S_OK;
}

status_t Nvmestore_session::get(uint64_t hashkey,
                                void*&   out_value,
                                size_t&  out_value_len)
{
  size_t blk_sz = _blk_manager->blk_sz();

  auto  root = _root;
  auto& pop  = _pop;

  TOID(struct block_range) blk_info;
  // TODO: can write to a shadowed copy
  try {
    blk_info = hm_tx_get(pop, D_RW(root)->map, hashkey);
    if (OID_IS_NULL(blk_info.oid)) return IKVStore::E_KEY_NOT_FOUND;

    auto val_len = D_RO(blk_info)->size;
    auto lba     = D_RO(blk_info)->lba_start;

#ifdef USE_ASYNC
    uint64_t tag = D_RO(blk_info)->last_tag;
    while (!blk_dev->check_completion(tag))
      cpu_relax(); /* check the last completion, TODO: check each time makes the
                      get slightly slow () */
#endif
    PDBG("prepare to read lba %d with length %d, key %lx", lba, val_len,
         hashkey);

    may_ajust_io_mem(val_len);
    size_t nr_io_blocks = (val_len + blk_sz - 1) / blk_sz;

    _blk_manager->do_block_io(nvmestore::BLOCK_IO_READ, _io_mem, lba, nr_io_blocks);

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

status_t Nvmestore_session::get_direct(uint64_t hashkey,
                                       void*    out_value,
                                       size_t&  out_value_len, buffer_t * memory_handle)
{
  size_t blk_sz = _blk_manager->blk_sz();

  auto root = _root;
  auto pop  = _pop;

  TOID(struct block_range) blk_info;
  try {
    cpu_time_t start = rdtsc();
    blk_info         = hm_tx_get(pop, D_RW(root)->map, hashkey);
    if (OID_IS_NULL(blk_info.oid)) return IKVStore::E_KEY_NOT_FOUND;

    auto val_len = D_RO(blk_info)->size;
    auto lba     = D_RO(blk_info)->lba_start;

    cpu_time_t cycles_for_hm = rdtsc() - start;

    PLOG("checked hxmap read latency took %ld cycles (%f usec) per hm access",
         cycles_for_hm, cycles_for_hm / 2400.0f);

#ifdef USE_ASYNC
    uint64_t tag = D_RO(blk_info)->last_tag;
    while (!blk_dev->check_completion(tag))
      cpu_relax(); /* check the last completion, TODO: check each time makes the
                      get slightly slow () */
#endif

    PDBG("prepare to read lba % lu with length %lu", lba, val_len);
    assert(out_value);

    io_buffer_t mem;

    if(memory_handle){ // external memory
      /* TODO: they are not nessarily equal, it memory is registered from outside */
      if(out_value < memory_handle->start_vaddr()){
        throw General_exception("out_value is not registered");
      }
      if(val_len > (size_t)(memory_handle->start_vaddr())){
        throw General_exception("registered memory is not big enough");
      }

      size_t offset = (size_t)out_value - (size_t)(memory_handle->start_vaddr());
      mem = memory_handle->io_mem() + offset;
    }
    else{
      mem = reinterpret_cast<io_buffer_t>(out_value);
    }

    assert(mem);

    size_t nr_io_blocks = (val_len + blk_sz - 1) / blk_sz;

    start = rdtsc();

    _blk_manager->do_block_io(nvmestore::BLOCK_IO_READ, mem, lba, nr_io_blocks);

    cpu_time_t cycles_for_iop = rdtsc() - start;
    PDBG("prepare to read lba % lu with nr_blocks %lu", lba, nr_io_blocks);
    PDBG("checked read latency took %ld cycles (%f usec) per IOP",
         cycles_for_iop, cycles_for_iop / 2400.0f);
    out_value_len = val_len;
  }
  catch (...) {
    throw General_exception("hm_tx_get failed unexpectedly");
  }
  return S_OK;
}

status_t Nvmestore_session::lock(uint64_t    hashkey,
                                 lock_type_t type,
                                 void*&      out_value,
                                 size_t&     out_value_len)
{
  auto& root           = _root;
  auto& pop            = _pop;
  int   operation_type = nvmestore::BLOCK_IO_NOP;

  size_t blk_sz = _blk_manager->blk_sz();
  auto   pool   = reinterpret_cast<uint64_t>(this);

  TOID(struct block_range) blk_info;
  try {
    blk_info = hm_tx_get(pop, D_RW(root)->map, hashkey);

    if (!OID_IS_NULL(blk_info.oid)) {
#ifdef USE_ASYNC
      /* there might be pending async write for this object */
      uint64_t tag = D_RO(blk_info)->last_tag;
      while (!blk_dev->check_completion(tag))
        cpu_relax(); /* check the last completion */
#endif
      operation_type = nvmestore::BLOCK_IO_READ;
    }
    else {
      /* TODO: need to create new object and continue */
      if (!out_value_len) {
        throw General_exception(
            "%s: Need value length to lock a unexsiting object", __func__);
      }
      alloc_new_object(hashkey, out_value_len, blk_info);
    }

    if (type == IKVStore::STORE_LOCK_READ) {
      if (!p_state_map->state_get_read_lock(pool, D_RO(blk_info)->handle))
        throw General_exception("%s: unable to get read lock", __func__);
    }
    else {
      if (!p_state_map->state_get_write_lock(pool, D_RO(blk_info)->handle))
        throw General_exception("%s: unable to get write lock", __func__);
    }

    auto handle    = D_RO(blk_info)->handle;
    auto value_len = D_RO(blk_info)->size;  // the length allocated before
    auto lba       = D_RO(blk_info)->lba_start;

    /* fetch the data to block io mem */
    size_t      nr_io_blocks = (value_len + blk_sz - 1) / blk_sz;
    io_buffer_t mem          = _blk_manager->allocate_io_buffer(
        nr_io_blocks * blk_sz, 4096, Component::NUMA_NODE_ANY);

    _blk_manager->do_block_io(operation_type, mem, lba, nr_io_blocks);

    get_locked_regions().emplace(
        mem, hashkey);  // TODO: can be placed in another place
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

  return S_OK;
}

status_t Nvmestore_session::unlock(uint64_t key_handle)
{
  auto        root    = _root;
  auto        pop     = _pop;
  auto        pool    = reinterpret_cast<uint64_t>(this);
  io_buffer_t mem     = reinterpret_cast<io_buffer_t>(key_handle);
  uint64_t    hashkey = get_locked_regions().at(mem);

  size_t blk_sz = _blk_manager->blk_sz();

  TOID(struct block_range) blk_info;
  try {
    blk_info = hm_tx_get(pop, D_RW(root)->map, (uint64_t) hashkey);
    if (OID_IS_NULL(blk_info.oid)) return IKVStore::E_KEY_NOT_FOUND;

    auto val_len = D_RO(blk_info)->size;
    auto lba     = D_RO(blk_info)->lba_start;

    size_t nr_io_blocks = (val_len + blk_sz - 1) / blk_sz;

    /*flush and release iomem*/
#ifdef USE_ASYNC
    uint64_t tag             = blk_dev->async_write(mem, 0, lba, nr_io_blocks);
    D_RW(blk_info)->last_tag = tag;
#else
    _blk_manager->do_block_io(nvmestore::BLOCK_IO_WRITE, mem, lba,
                              nr_io_blocks);
#endif

    PDBG("[nvmestore_session]: freeing io mem at %p", (void*) mem);
    _blk_manager->free_io_buffer(mem);

    /*release the lock*/
    p_state_map->state_unlock(pool, D_RO(blk_info)->handle);

    PDBG("NVME_store: released the lock");
  }
  catch (...) {
    throw General_exception("hm_tx_get failed unexpectedly or iomem not found");
  }
  return S_OK;
}

using open_session_t = Nvmestore_session;

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
                       const std::string& pm_path)
    : _pm_path(pm_path), _blk_manager(pci, pm_path)
{
  if (_pm_path.back() != '/') _pm_path += "/";

  PLOG("NVMESTORE: chunk size in blocks: %lu", CHUNK_SIZE_IN_BLOCKS);
  PLOG("PMEMOBJ_MAX_ALLOC_SIZE: %lu MB", REDUCE_MB(PMEMOBJ_MAX_ALLOC_SIZE));
}

NVME_store::~NVME_store() { if(option_DEBUG) PLOG("delete NVME store"); }

IKVStore::pool_t NVME_store::create_pool(const std::string& name,
                                         const size_t       size,
                                         unsigned int       flags,
                                         uint64_t           args)
{
  PMEMobjpool* pop = nullptr;  // pool to allocate all mapping
  int          ret = 0;

  size_t max_sz_hxmap = MB(500);  // this can fit 1M objects (block_range_t)

  // TODO: need to check size
  // TODO: pass prefix (pm_path) into nvmestore component config
  const std::string& fullpath = _pm_path + name;

  PINF("[NVME_store]::create_pool fullpath=%s name=%s", fullpath.c_str(),
       name.c_str());

  /* open existing pool */
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

  assert(ret == 0);

  if (hm_tx_check(pop, D_RO(root)->map))
    throw General_exception("hm_tx_check failed unexpectedly");

  open_session_t* session = new open_session_t(
      root, pop, size, fullpath, DEFAULT_IO_MEM_SIZE, &_blk_manager, &_sm);

  g_sessions.insert(session);

  return reinterpret_cast<uint64_t>(session);
}

IKVStore::pool_t NVME_store::open_pool(const std::string& name,
                                       unsigned int       flags)
{
  PMEMobjpool* pop;               // pool to allocate all mapping
  size_t max_sz_hxmap = MB(500);  // this can fit 1M objects (block_range_t)

  PINF("NVME_store::open_pool name=%s", name.c_str());

  const std::string& fullpath = _pm_path + name;

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

  g_sessions.insert(session);

  return reinterpret_cast<uint64_t>(session);
}

status_t NVME_store::close_pool(pool_t pid)
{
  open_session_t* session = reinterpret_cast<open_session_t*>(pid);

  if (g_sessions.find(session) == g_sessions.end()) {
    return E_INVAL;
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

  if (access(fullpath.c_str(), F_OK) != 0) {
    PWRN("nvmestore: pool doesn't exsit!");
    return E_POOL_NOT_FOUND;
  }

  // TODO should clean the blk_allocator and blk dev (reference) here?
  //_blk_alloc->resize(0, 0);

  if (pmempool_rm(fullpath.c_str(), 0))
    throw General_exception("unable to delete pool (%s)", fullpath.c_str());

  PLOG("pool deleted: %s", session->path.c_str());
  return S_OK;
}

/**
 * Create a entry in the pool and allocate space
 *
 * @param session
 * @param value_len
 * @param out_blkmeta [out] block mapping info of this obj
 *
 * TODO This allocate memory using regions */
void Nvmestore_session::alloc_new_object(uint64_t hashkey,
                                         size_t   value_len,
                                         TOID(struct block_range) & out_blkmeta)
{
  auto& root        = this->_root;
  auto& pop         = this->_pop;
  auto& blk_manager = this->_blk_manager;

  size_t blk_size     = blk_manager->blk_sz();
  size_t nr_io_blocks = (value_len + blk_size - 1) / blk_size;

  void* handle;

  // transaction also happens in here
  uint64_t lba = _blk_manager->alloc_blk_region(nr_io_blocks, &handle);

  PDBG("write to lba %lu with length %lu, key %lx", lba, value_len, hashkey);

  auto& blk_info = out_blkmeta;
  TX_BEGIN(pop)
  {
    /* allocate memory for entry - range added to tx implicitly? */

    // get the available range from allocator
    blk_info = TX_ALLOC(struct block_range, sizeof(struct block_range));

    D_RW(blk_info)->lba_start = lba;
    D_RW(blk_info)->size      = value_len;
    D_RW(blk_info)->handle    = handle;

    /* insert into HT */
    int rc;
    if ((rc = hm_tx_insert(pop, D_RW(root)->map, hashkey, blk_info.oid))) {
      if (rc == 1)
        throw General_exception("inserting same key");
      else
        throw General_exception("hm_tx_insert failed unexpectedly (rc=%d)", rc);
    }
  }
  TX_ONABORT
  {
    // TODO: free blk_info
    throw General_exception("TX abort (%s) during nvmeput", pmemobj_errormsg());
  }
  TX_END
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
    throw API_exception("NVME_store::put invalid pool identifier");

  uint64_t hashkey = CityHash64(key.c_str(), key.length());
  return session->put(hashkey, value, value_len, flags);
}

status_t NVME_store::get(const pool_t       pool,
                         const std::string& key,
                         void*&             out_value,
                         size_t&            out_value_len)
{
  open_session_t* session = reinterpret_cast<open_session_t*>(pool);

  if (g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store::put invalid pool identifier");
  uint64_t hashkey = CityHash64(key.c_str(), key.length());

  return session->get(hashkey, out_value, out_value_len);
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
  uint64_t hashkey = CityHash64(key.c_str(), key.length());

  return session->get_direct(hashkey, out_value, out_value_len, reinterpret_cast<buffer_t *>(handle));

  // return reinterpret_cast<Component::IKVStore::key_t>(out_value);
}

static_assert(sizeof(IKVStore::memory_handle_t) == sizeof(io_buffer_t),
              "cast may not work");

/*
 * Only used for the case when memory is pinned/aligned but not from spdk, e.g.
 * cudadma should be 2MB aligned in both phsycial and virtual*/
IKVStore::memory_handle_t NVME_store::register_direct_memory(void*  vaddr,
                                                             size_t len)
{
  memset(vaddr, 0, len);

  addr_t phys_addr = xms_get_phys(vaddr);
  io_buffer_t io_mem  = _blk_manager.register_memory_for_io(vaddr, phys_addr, len);
  buffer_t *buffer = new buffer_t(len, io_mem, vaddr);

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
  buffer_t *buffer = reinterpret_cast<buffer_t *>(handle);
  _blk_manager.unregister_memory_for_io(buffer->start_vaddr(), buffer->length());
  delete buffer;
  return S_OK;
}

// status_t NVME_store::allocate(const pool_t pool,
//                               const std::string& key,
//                               const size_t nbytes,
//                               uint64_t& out_key_hash)
// {
//   open_session_t * session = get_session(pool);
//   auto& root = session->root;
//   auto& pop = session->pop;

//   auto& blk_alloc = session->_blk_alloc;
//   auto& blk_dev = session->_blk_dev;

//   uint64_t key_hash = CityHash64(key.c_str(), key.length());

//   void * handle;

//   /* check to see if key already exists */
//   /*if(hm_tx_lookup(pop, d_ro(root)->map, key_hash))*/
//   /*return e_key_exists;*/

//   size_t nr_io_blocks = (nbytes+ blk_sz -1)/blk_sz;

//   // transaction also happens in here
//   lba_t lba = blk_alloc->alloc(nr_io_blocks, &handle);

//   TOID(struct block_range) blk_info;

//   TX_BEGIN(pop) {

//     /* allocate memory for entry - range added to tx implicitly? */

//     //get the available range from allocator
//     blk_info = TX_ALLOC(struct block_range, sizeof(struct block_range));

//     D_RW(blk_info)->lba_start = lba;
//     D_RW(blk_info)->size = nbytes;
//     D_RW(blk_info)->handle = handle;
// #ifdef USE_ASYNC
//     D_RW(blk_info)->last_tag = 0;
// #endif

//     /* insert into HT */
//     int rc;
//     if((rc = hm_tx_insert(pop, D_RW(root)->map, key_hash, blk_info.oid))) {
//       if(rc == 1)
//         return E_ALREADY_EXISTS;
//       else throw General_exception("hm_tx_insert failed unexpectedly
//       (rc=%d)", rc);
//     }
//   }
//   TX_ONABORT {
//     //TODO: free blk_range
//     throw General_exception("TX abort (%s)", pmemobj_errormsg());
//   }
//   TX_END

//     out_key_hash = key_hash;

//   return S_OK;
// }

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

  uint64_t hashkey = CityHash64(key.c_str(), key.length());

  session->lock(hashkey, type, out_value, out_value_len);
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

status_t NVME_store::erase(const pool_t pool, const std::string& key)
{
  open_session_t* session = get_session(pool);

  uint64_t key_hash = CityHash64(key.c_str(), key.length());
  return session->erase(key_hash);
}

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

  Component::IKVStore* obj = static_cast<Component::IKVStore*>(new NVME_store(
      params["owner"], params["name"], params["pci"], params["pm_path"]));
  obj->add_ref();
  return obj;
}
