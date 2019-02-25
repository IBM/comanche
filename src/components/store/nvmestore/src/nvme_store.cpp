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

#include <iostream>
#include <set>
#include <libpmemobj.h>
#include <libpmempool.h>
#include <libpmemobj/base.h>

#include <stdio.h>
#include <api/kvstore_itf.h>
#include <city.h>
#include <boost/filesystem.hpp>
#include <common/cycles.h>

#include <core/xms.h>
#include <component/base.h>
#include <api/components.h>
#include <api/block_itf.h>

#include "state_map.h"


extern "C"
{
#include "hashmap_tx.h"
}

//#define USE_ASYNC

using namespace Component;


struct store_root_t
{
  TOID(struct hashmap_tx) map; //name mappings
  size_t pool_size;
};
//TOID_DECLARE_ROOT(struct store_root_t);

struct open_session_t
{
  TOID(struct store_root_t) root;
  PMEMobjpool *             pop; // the pool for mapping
  size_t                    pool_size;
  std::string               path;
  uint64_t io_mem; // io memory for lock/unload TODO: this should be thead-safe, this should be large enough store the allocated value
  size_t io_mem_size; // io memory size which will an object;
  Component::IBlock_device *_blk_dev;
  Component::IBlock_allocator *_blk_alloc;
  std::unordered_map<uint64_t, io_buffer_t> _locked_regions;
};

struct tls_cache_t {
  open_session_t * session;
};

static __thread tls_cache_t tls_cache = { nullptr };
std::set<open_session_t*> g_sessions;

static open_session_t * get_session(IKVStore::pool_t pid) //open_session_t * session)
{
  open_session_t * session = reinterpret_cast<struct open_session_t*>(pid);
  if(session == tls_cache.session && session != nullptr) return session;

  if(g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store:: invalid pool identifier");

  return session;
}

static int check_pool(const char * path)
{
  PLOG("check_pool: %s", path);
  PMEMpoolcheck *ppc;
  struct pmempool_check_status * status;

  struct pmempool_check_args args;
  args.path = path;
  args.backup_path = NULL;
  args.pool_type = PMEMPOOL_POOL_TYPE_DETECT;
  args.flags =
    PMEMPOOL_CHECK_FORMAT_STR |
    PMEMPOOL_CHECK_REPAIR |
    PMEMPOOL_CHECK_VERBOSE;

  if((ppc = pmempool_check_init(&args, sizeof(args))) == NULL) {
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
                       const std::string& pm_path) : _pm_path(pm_path)
{
  PLOG("NVMESTORE: chunk size in blocks: %lu", CHUNK_SIZE_IN_BLOCKS);
  PLOG("PMEMOBJ_MAX_ALLOC_SIZE: %lu MB", REDUCE_MB(PMEMOBJ_MAX_ALLOC_SIZE));

  /* Note: see note in open_block_device for failure info */
  status_t ret = open_block_device(pci, _blk_dev);
  if(S_OK != ret){
    throw General_exception("failed (%d) to open block device at pci %s\n", ret, pci.c_str());
  }

  ret = open_block_allocator(_blk_dev, _blk_alloc);

  if(S_OK != ret){
    throw General_exception("failed (%d) to open block block allocator for device at pci %s\n", ret, pci.c_str());
  }

  PDBG("NVME_store: using block device %p with allocator %p", _blk_dev, _blk_alloc);
}

NVME_store::~NVME_store()
{

  PINF("delete NVME store");
  assert(_blk_dev);
  assert(_blk_alloc);
  _blk_alloc->release_ref();
  _blk_dev->release_ref();
}

IKVStore::pool_t NVME_store::create_pool(const std::string& path,
                                         const std::string& name,
                                         const size_t size,
                                         unsigned int flags,
                                         uint64_t args)
{
  PMEMobjpool *pop = nullptr; //pool to allocate all mapping
  int ret =0;

  size_t max_sz_hxmap = MB(500); // this can fit 1M objects (block_range_t)

  // TODO: need to check size
  // TODO: pass prefix (pm_path) into nvmestore component config
  std::string fullpath = "/mnt/pmem0/";
  
  if(path[path.length()-1]!='/')
    fullpath += path + "/" + name;
  else
    fullpath += path + name;

  PINF("NVME_store::create_pool fullpath=%s name=%s", fullpath.c_str(), name.c_str());
  
  /* open existing pool */
  pop = pmemobj_open(fullpath.c_str(), POBJ_LAYOUT_NAME(nvme_store));

  if(!pop) {
    PLOG("creating new pool: %s", name.c_str());

    boost::filesystem::path p(fullpath);
    boost::filesystem::create_directories(p.parent_path());

    pop = pmemobj_create(fullpath.c_str(), POBJ_LAYOUT_NAME(nvme_store), max_sz_hxmap, 0666);
  }

  if(not pop)
    throw General_exception("failed to create or open pool (%s)", pmemobj_errormsg());

  /* see: https://github.com/pmem/pmdk/blob/stable-1.4/src/examples/libpmemobj/map/kv_server.c */
  assert(pop);
  TOID(struct store_root_t) root = POBJ_ROOT(pop, struct store_root_t);
  assert(!TOID_IS_NULL(root));

  if(D_RO(root)->map.oid.off == 0) {

    /* create hash table if it does not exist */
    TX_BEGIN(pop) {
      if(hm_tx_create(pop, &D_RW(root)->map, nullptr))
        throw General_exception("hm_tx_create failed unexpectedly");
      D_RW(root)->pool_size = size;
    }
    TX_ONABORT {
      ret = -1;
    } TX_END
  }

  assert(ret == 0);

  if(hm_tx_check(pop, D_RO(root)->map))
    throw General_exception("hm_tx_check failed unexpectedly");

  struct open_session_t * session = new open_session_t;
  session->root = root;
  session->pop = pop;
  session->pool_size = size;
  session->path = fullpath;
  session->io_mem_size = DEFAULT_IO_MEM_SIZE;
  session->io_mem = _blk_dev->allocate_io_buffer(DEFAULT_IO_MEM_SIZE, 4096,Component::NUMA_NODE_ANY);
  session->_blk_dev = _blk_dev;
  session->_blk_alloc = _blk_alloc;
  g_sessions.insert(session);

  return reinterpret_cast<uint64_t>(session);
}

IKVStore::pool_t NVME_store::open_pool(const std::string& path,
                                       const std::string& name,
                                       unsigned int flags)
{
  PMEMobjpool *pop; //pool to allocate all mapping
  size_t max_sz_hxmap = MB(500); // this can fit 1M objects (block_range_t)

  PINF("NVME_store::open_pool path=%s name=%s", path.c_str(), name.c_str());

  std::string fullpath;

  if(path[path.length()-1]!='/')
    fullpath = path + "/" + name;
  else
    fullpath = path + name;

  /* if trying to open a unclosed pool!*/
  for(auto iter : g_sessions){
    if(iter->path == fullpath){
      PWRN("nvmestore: try to reopen a pool!");
      return reinterpret_cast<uint64_t>(iter);
    }
  }

  if (access(fullpath.c_str(), F_OK) != 0) {
    throw General_exception("nvmestore: pool not existing at path %s", fullpath.c_str());
  }
  else {
    PLOG("Opening existing Pool: %s", name.c_str());

    if(check_pool(fullpath.c_str()) != 0)
      throw General_exception("pool check failed");

    pop = pmemobj_open(fullpath.c_str(),POBJ_LAYOUT_NAME(nvme_store));
    if(not pop)
      throw General_exception("failed to re-open pool - %s\n", pmemobj_errormsg());
  }

  TOID(struct store_root_t) root = POBJ_ROOT(pop, struct store_root_t);
  assert(!TOID_IS_NULL(root));

  assert(D_RO(root)->map.oid.off != 0);
  /*TODO; in caffe workload the poolsize is not persist somehow*/
  if(D_RO(root)->pool_size == 0){
    PWRN("nvmestore: pool size is ZERO!");
  }
  PLOG("Using existing root, pool size =  %lu:", D_RO(root)->pool_size);
  if(hm_tx_init(pop, D_RW(root)->map))
    throw General_exception("hm_tx_init failed unexpectedly");

  struct open_session_t * session = new open_session_t;
  session->root = root;
  session->pop = pop;
  session->pool_size = D_RO(root)->pool_size;
  session->path = fullpath;
  session->io_mem_size = DEFAULT_IO_MEM_SIZE;
  session->io_mem = _blk_dev->allocate_io_buffer(DEFAULT_IO_MEM_SIZE, 4096,Component::NUMA_NODE_ANY);
  g_sessions.insert(session);

  return reinterpret_cast<uint64_t>(session);
}

status_t NVME_store::close_pool(pool_t pid)
{
  struct open_session_t * session = reinterpret_cast<struct open_session_t*>(pid);

  if(g_sessions.find(session) == g_sessions.end()){
    return E_INVAL;
  }

  io_buffer_t mem = session->io_mem;
  if(mem)
    _blk_dev->free_io_buffer(mem);

  pmemobj_close(session->pop);

  g_sessions.erase(session);
  PLOG("NVME_store::closed pool (%lx)", pid);
  return S_OK;
}

status_t NVME_store::delete_pool(const pool_t pid)
{
  struct open_session_t * session = reinterpret_cast<struct open_session_t*>(pid);

  if(g_sessions.find(session) == g_sessions.end())
    return E_INVAL;

  g_sessions.erase(session);
  pmemobj_close(session->pop);
  //TODO should clean the blk_allocator and blk dev (reference) here?
  //_blk_alloc->resize(0, 0);

  if(pmempool_rm(session->path.c_str(), 0))
    throw General_exception("unable to delete pool (%p)", pid);

  PLOG("pool deleted: %s", session->path.c_str());
  return S_OK;
}

/* Create a entry in the pool and allocate space
 *
 * @param session
 * @param value_len
 * @param out_blkmeta [out] block mapping info of this obj
 *
 * TODO This allocate memory using regions */
static int __alloc_new_object(struct open_session_t *session, uint64_t hashkey, size_t value_len, TOID(struct block_range) &out_blkmeta){

  auto& root = session->root;
  auto& pop = session->pop;

  size_t blk_size = 4096; //TODO: need to detect
  size_t nr_io_blocks = (value_len+ blk_size -1)/blk_size;

  void * handle;

  // transaction also happens in here
  uint64_t lba = session->_blk_alloc->alloc(nr_io_blocks, &handle);

  PDBG("write to lba %lu with length %lu, key %lx",lba, value_len, hashkey);

  auto &blk_info = out_blkmeta;
  TX_BEGIN(pop) {

    /* allocate memory for entry - range added to tx implicitly? */

    //get the available range from allocator
    blk_info = TX_ALLOC(struct block_range, sizeof(struct block_range));

    D_RW(blk_info)->lba_start = lba;
    D_RW(blk_info)->size = value_len;
    D_RW(blk_info)->handle = handle;

    /* insert into HT */
    int rc;
    if((rc = hm_tx_insert(pop, D_RW(root)->map, hashkey, blk_info.oid))) {
      if(rc == 1)
        throw General_exception("inserting same key");
      else throw General_exception("hm_tx_insert failed unexpectedly (rc=%d)", rc);
    }
  }
  TX_ONABORT {
    //TODO: free blk_info
    throw General_exception("TX abort (%s) during nvmeput", pmemobj_errormsg());
  }
  TX_END

  return 0;
}

/*
 * when using NVMe, only insert the block range descriptor into the mapping
 */
status_t NVME_store::put(IKVStore::pool_t pool,
                         const std::string& key,
                         const void * value,
                         size_t value_len)
{
  struct open_session_t * session = reinterpret_cast<struct open_session_t*>(pool);

  if(g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store::put invalid pool identifier");

  auto& root = session->root;
  auto& pop = session->pop;

  auto& blk_alloc = session->_blk_alloc;
  auto& blk_dev = session->_blk_dev;

  uint64_t hashkey = CityHash64(key.c_str(), key.length());
  TOID(struct block_range) blkmeta; // block mapping of this obj

  /* TODO: increase IO buffer sizes when value size is large*/
  if(value_len > session->io_mem_size){
    throw General_exception("Object size larger than MB(8)!");
  }

  if(hm_tx_lookup(pop, D_RO(root)->map, hashkey))
    return E_KEY_EXISTS;

  __alloc_new_object(session, hashkey, value_len, blkmeta);

  memcpy(blk_dev->virt_addr(session->io_mem), value, value_len); /* for the moment we have to memcpy */

#ifdef USE_ASYNC
#error("use_sync is deprecated")
  // TODO: can the free be triggered by callback?
  uint64_t tag = blk_dev->async_write(session->io_mem, 0, lba, nr_io_blocks);
  D_RW(blk_info)->last_tag = tag;
#else
  auto nr_io_blocks = (value_len+ BLOCK_SIZE -1)/BLOCK_SIZE;
  do_block_io(blk_dev, BLOCK_IO_WRITE, session->io_mem, D_RO(blkmeta)->lba_start, nr_io_blocks);
#endif

  _cnt_elem_map[pool] ++;

  return S_OK;
}

status_t NVME_store::get(const pool_t pool,
                         const std::string& key,
                         void*& out_value,
                         size_t& out_value_len)
{
  struct open_session_t * session = reinterpret_cast<struct open_session_t*>(pool);

  if(g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store::put invalid pool identifier");

  auto& root = session->root;
  auto& pop = session->pop;
  auto mem = session->io_mem;

  auto& blk_alloc = session->_blk_alloc;
  auto& blk_dev = session->_blk_dev;

  uint64_t hashkey = CityHash64(key.c_str(), key.length());

  TOID(struct block_range) blk_info;
  // TODO: can write to a shadowed copy
  try {
    blk_info = hm_tx_get(pop, D_RW(root)->map, hashkey);
    if(OID_IS_NULL(blk_info.oid))
      return E_KEY_NOT_FOUND;

    auto val_len = D_RO(blk_info)->size;
    auto lba = D_RO(blk_info)->lba_start;

#ifdef USE_ASYNC
    uint64_t tag = D_RO(blk_info)->last_tag;
    while(!blk_dev->check_completion(tag)) cpu_relax(); /* check the last completion, TODO: check each time makes the get slightly slow () */
#endif
    PDBG("prepare to read lba %d with length %d, key %lx", lba, val_len, hashkey);
    size_t nr_io_blocks = (val_len+ BLOCK_SIZE -1)/BLOCK_SIZE;

    do_block_io(blk_dev, BLOCK_IO_READ, mem, lba, nr_io_blocks);

    out_value = malloc(val_len);
    assert(out_value);
    memcpy(out_value, blk_dev->virt_addr(mem), val_len);
    out_value_len = val_len;
  }
  catch(...) {
    throw General_exception("hm_tx_get failed unexpectedly");
  }
  return S_OK;
}

status_t NVME_store::get_direct(const pool_t pool,
                                const std::string& key,
                                void* out_value,
                                size_t& out_value_len,
                                Component::IKVStore::memory_handle_t handle)
{
  struct open_session_t * session = reinterpret_cast<struct open_session_t*>(pool);

  if(g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store::put invalid pool identifier");

  auto& root = session->root;
  auto& pop = session->pop;

  auto& blk_alloc = session->_blk_alloc;
  auto& blk_dev = session->_blk_dev;

  uint64_t hashkey = CityHash64(key.c_str(), key.length());

  TOID(struct block_range) blk_info;
  try {
    cpu_time_t start = rdtsc() ;
    blk_info = hm_tx_get(pop, D_RW(root)->map, hashkey);
    if(OID_IS_NULL(blk_info.oid))
      return E_KEY_NOT_FOUND;

    auto val_len = D_RO(blk_info)->size;
    auto lba = D_RO(blk_info)->lba_start;

    cpu_time_t cycles_for_hm = rdtsc() - start;

    PLOG("checked hxmap read latency took %ld cycles (%f usec) per hm access", cycles_for_hm,  cycles_for_hm / 2400.0f);

#ifdef USE_ASYNC
    uint64_t tag = D_RO(blk_info)->last_tag;
    while(!blk_dev->check_completion(tag)) cpu_relax(); /* check the last completion, TODO: check each time makes the get slightly slow () */
#endif

    PDBG("prepare to read lba % lu with length %lu", lba, val_len);
    assert(out_value);

    /* TODO: safe? */
    io_buffer_t mem = reinterpret_cast<Component::io_buffer_t>(out_value);

    assert(mem);

    size_t nr_io_blocks = (val_len+ BLOCK_SIZE -1)/BLOCK_SIZE;

    start = rdtsc() ;

    do_block_io(blk_dev, BLOCK_IO_READ, mem, lba, nr_io_blocks);

    cpu_time_t cycles_for_iop = rdtsc() - start;
    PDBG("prepare to read lba % lu with nr_blocks %lu", lba, nr_io_blocks);
    PDBG("checked read latency took %ld cycles (%f usec) per IOP", cycles_for_iop,  cycles_for_iop / 2400.0f);
    out_value_len = val_len;
  }
  catch(...) {
    throw General_exception("hm_tx_get failed unexpectedly");
  }
  return S_OK;
}

static_assert(sizeof(IKVStore::memory_handle_t) == sizeof(io_buffer_t), "cast may not work");
/*
 * Only used for the case when memory is pinned/aligned but not from spdk, e.g. cudadma
 * should be 2MB aligned in both phsycial and virtual*/
IKVStore::memory_handle_t NVME_store::register_direct_memory(void * vaddr, size_t len){
  addr_t phys_addr; // physical address
  io_buffer_t handle = 0;;

  phys_addr = xms_get_phys(vaddr);
  handle = _blk_dev->register_memory_for_io(vaddr, phys_addr, len);

  auto result = reinterpret_cast<IKVStore::memory_handle_t>(handle);
  /* save this this registration */
  if(handle)
    PINF("Register vaddr %p with paddr %lu, handle %lu", vaddr, phys_addr, handle );
  else
    PERR("%s: register user allocated memory failed", __func__);

  return result;
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

//   size_t nr_io_blocks = (nbytes+ BLOCK_SIZE -1)/BLOCK_SIZE;

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
//       else throw General_exception("hm_tx_insert failed unexpectedly (rc=%d)", rc);
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

/* For nvmestore, data is not necessarily in main memory.
 * Lock will allocate iomem and load data from nvme first.
 * Unlock will will free it
 *
 * Note: nvmestore does not support multiple read locks on the same key by the same thread.
 */
IKVStore::key_t NVME_store::lock(const pool_t pool,
                                 const std::string& key,
                                 lock_type_t type,
                                 void*& out_value,
                                 size_t& out_value_len)
{
  open_session_t * session = get_session(pool);

  auto& root = session->root;
  auto& pop = session->pop;
  int operation_type= BLOCK_IO_NOP;

  auto& blk_dev = session->_blk_dev;

  uint64_t hashkey = CityHash64(key.c_str(), key.length());

  TOID(struct block_range) blk_info;
  try {
    blk_info = hm_tx_get(pop, D_RW(root)->map, hashkey);

    if(!OID_IS_NULL(blk_info.oid)){
#ifdef USE_ASYNC
      /* there might be pending async write for this object */
      uint64_t tag = D_RO(blk_info)->last_tag;
      while(!blk_dev->check_completion(tag)) cpu_relax(); /* check the last completion */
#endif
      operation_type = BLOCK_IO_READ;
    }
    else{
      /* TODO: need to create new object and continue */
      if(!out_value_len){
          throw General_exception("%s: Need value length to lock a unexsiting object", __func__);
      }
      __alloc_new_object(session, hashkey,out_value_len, blk_info);
      _cnt_elem_map[pool]++;
    }

    if(type == IKVStore::STORE_LOCK_READ) {
      if(!_sm.state_get_read_lock(pool, D_RO(blk_info)->handle))
        throw General_exception("%s: unable to get read lock", __func__);
    }
    else {
      if(!_sm.state_get_write_lock(pool, D_RO(blk_info)->handle))
        throw General_exception("%s: unable to get write lock", __func__);
    }

    auto handle = D_RO(blk_info)->handle;
    auto value_len = D_RO(blk_info)->size; // the length allocated before
    auto lba = D_RO(blk_info)->lba_start;

    /* fetch the data to block io mem */
    size_t nr_io_blocks = (value_len + BLOCK_SIZE -1)/BLOCK_SIZE;
    io_buffer_t mem = blk_dev->allocate_io_buffer(nr_io_blocks*4096, 4096,Component::NUMA_NODE_ANY);

    do_block_io(blk_dev, operation_type, mem, lba, nr_io_blocks);

    session->_locked_regions.emplace(hashkey, mem); //TODO: can be placed in another place

    /* set output values */
    out_value = blk_dev->virt_addr(mem);
    out_value_len = value_len;
  }
  catch(...){
    PERR("NVME_store: lock failed");
  }

  PINF("NVME_store: obtained the lock");
  return reinterpret_cast<Component::IKVStore::key_t>(hashkey);
}

/*
 *
 * For nvmestore, data is not necessarily in main memory.
 * Lock will allocate iomem and load data from nvme first.
 * Unlock will will free it
 */
status_t NVME_store::unlock(const pool_t pool,
                            key_t key_hash)
{
  open_session_t * session = get_session(pool);

  auto& root = session->root;
  auto& pop = session->pop;


  auto& blk_dev = session->_blk_dev;

  TOID(struct block_range) blk_info;
  try {
    blk_info = hm_tx_get(pop, D_RW(root)->map, (uint64_t) key_hash);
    if(OID_IS_NULL(blk_info.oid))
      return E_KEY_NOT_FOUND;

    auto val_len = D_RO(blk_info)->size;
    auto lba = D_RO(blk_info)->lba_start;

    size_t nr_io_blocks = (val_len + BLOCK_SIZE -1)/BLOCK_SIZE;
    io_buffer_t mem = session->_locked_regions.at((uint64_t)key_hash);

    /*flush and release iomem*/
#ifdef USE_ASYNC
    uint64_t tag = blk_dev->async_write(mem, 0, lba, nr_io_blocks);
    D_RW(blk_info)->last_tag = tag;
#else
    do_block_io(blk_dev, BLOCK_IO_WRITE, mem, lba, nr_io_blocks);
#endif

    blk_dev->free_io_buffer(mem);

    /*release the lock*/
    _sm.state_unlock(pool, D_RO(blk_info)->handle);

    PINF("NVME_store: released the lock");
  }
  catch(...) {
    throw General_exception("hm_tx_get failed unexpectedly or iomem not found");
  }

  return S_OK;
}

status_t NVME_store::apply(const pool_t pool,
                           const std::string& key,
                           std::function<void(void*,const size_t)> functor,
                           size_t object_size,
                           bool take_lock)
{

  void * data;
  size_t value_len = 0;
  key_t obj_key;

  if(take_lock){
    obj_key = lock(pool, key, IKVStore::STORE_LOCK_WRITE, data, object_size);
  }
  /* TODO FIX: for take_lock. if take_lock == TRUE then use a lock here */
  //  lock(pool, CityHash64(key.c_str(), key.length()),IKVStore::STORE_LOCK_READ, data, value_len);
  functor(data, object_size);

  if(take_lock){
    unlock(pool, obj_key);
  }
  return S_OK;

}


status_t NVME_store::erase(const pool_t pool,
                           const std::string& key)
{
  uint64_t key_hash = CityHash64(key.c_str(), key.length());
  open_session_t * session = get_session(pool);

  auto& root = session->root;
  auto& pop = session->pop;

  auto& blk_alloc = session->_blk_alloc;
  auto& blk_dev = session->_blk_dev;

  TOID(struct block_range) blk_info;

  try {
    blk_info = hm_tx_get(pop, D_RW(root)->map, key_hash);
    if(OID_IS_NULL(blk_info.oid))
      return E_KEY_NOT_FOUND;

    /* get hold of write lock to remove */
    if(!_sm.state_get_write_lock(pool, D_RO(blk_info)->handle))
      throw API_exception("unable to remove, value locked");

    blk_alloc->free(D_RO(blk_info)->lba_start, D_RO(blk_info)->handle);

    blk_info = hm_tx_remove(pop, D_RW(root)->map, key_hash); /* could be optimized to not re-lookup */
    if(OID_IS_NULL(blk_info.oid))
      throw API_exception("hm_tx_remove failed unexpectedly");

    _sm.state_remove(pool, D_RO(blk_info)->handle);
  }
  catch(...) {
    throw General_exception("hm_tx_remove failed unexpectedly");
  }

  _cnt_elem_map[pool]--;
  return S_OK;
}


/**
 * Factory entry point
 *
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == NVME_store_factory::component_id()) {
    return reinterpret_cast<void*>(new NVME_store_factory());
  }
  else return NULL;
}

void * NVME_store_factory::query_interface(Component::uuid_t& itf_uuid) {
  if(itf_uuid == Component::IKVStore_factory::iid()) {
    return this;
  }
  else return NULL; // we don't support this interface
}


void NVME_store_factory::unload() {
  delete this;
}

auto NVME_store_factory::create(const std::string& owner,
                                const std::string& name,
                                const std::string& pci) -> Component::IKVStore *
{
  if ( pci.size() != 7 || pci[2] != ':' || pci[5] != '.' )
  {
    PWRN("Parameter '%s' does not look like a PCI address", pci.c_str());
  }
  /* TODO, 3rd parameter to create should be a JSON string including pci address and pmem path */

  Component::IKVStore * obj = static_cast<Component::IKVStore*>(new NVME_store(owner, name, pci, "/mnt/pmem0"));
  obj->add_ref();
  return obj;
}

auto NVME_store_factory::create(unsigned,
                                const std::string& owner,
                                const std::string& name,
                                const std::string& pci) -> Component::IKVStore *
{
  return create(owner, name, pci);
}
