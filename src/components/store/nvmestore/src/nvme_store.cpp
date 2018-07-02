/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */
#include "nvme_store.h"

#include <iostream>
#include <set>
#include <libpmemobj.h>
#include <libpmempool.h>
#include <libpmemobj/base.h>

#include <stdio.h>
#include <api/kvstore_itf.h>
#include <common/city.h>
#include <boost/filesystem.hpp>

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
};
//TOID_DECLARE_ROOT(struct store_root_t);




struct open_session_t
{
  TOID(struct store_root_t) root;
  PMEMobjpool *             pop; // the pool for mapping
  size_t                    pool_size;
  std::string               path;
  uint64_t io_mem; // io memory for lock/unload TODO: this should be thead-safe
};

struct tls_cache_t {
  open_session_t * session;
};

static __thread tls_cache_t tls_cache = { nullptr };
std::set<open_session_t*> g_sessions;

static open_session_t * get_session(IKVStore::pool_t pid) //open_session_t * session) 
{
  open_session_t * session = reinterpret_cast<struct open_session_t*>(pid);
  if(session == tls_cache.session) return session;

  if(g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store::delete_pool invalid pool identifier");

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


NVME_store::NVME_store(const std::string owner,
                       const std::string name)
                       {
  PLOG("PMEMOBJ_MAX_ALLOC_SIZE: %lu MB", REDUCE_MB(PMEMOBJ_MAX_ALLOC_SIZE));

  init_block_device();
  init_block_allocator();

  PINF("NVME_store: using block device %p with allocator %p", _blk_dev, _blk_alloc);
}

NVME_store::~NVME_store()
{

  PINF("delete NVME store");
  assert(_blk_dev);
  assert(_blk_alloc);
  _blk_alloc->release_ref();
  _blk_dev->release_ref();
}

IKVStore::pool_t NVME_store::create_pool(const std::string path,
                                      const std::string name,
                                      const size_t size,
                                      unsigned int flags,
                                      uint64_t args)
{
  PMEMobjpool *pop; //pool to allocate all mapping
  
  PINF("NVME_store::create_pool path=%s name=%s", path.c_str(), name.c_str());

  size_t max_sz_hxmap = MB(200);

  // TODO: need to check size

  std::string fullpath;

  if(path[path.length()-1]!='/')
    fullpath = path + "/" + name;
  else
    fullpath = path + name;

  if (access(fullpath.c_str(), F_OK) != 0) {
    PLOG("Creating new Pool: %s", name.c_str());

    boost::filesystem::path p(fullpath);
    boost::filesystem::create_directories(p.parent_path());

    pop = pmemobj_create(fullpath.c_str(), POBJ_LAYOUT_NAME(nvme_store), max_sz_hxmap, 0666);
    if(not pop)
      throw General_exception("failed to create new pool - %s\n", pmemobj_errormsg());
  }
  else {
    PLOG("Opening existing Pool: %s", name.c_str());

    if(check_pool(fullpath.c_str()) != 0)
      throw General_exception("pool check failed");

    pop = pmemobj_open(fullpath.c_str(),POBJ_LAYOUT_NAME(nvme_store));
    if(not pop)
      throw General_exception("failed to re-open pool - %s\n", pmemobj_errormsg());
  }

  /* see: https://github.com/pmem/pmdk/blob/stable-1.4/src/examples/libpmemobj/map/kv_server.c */

  TOID(struct store_root_t) root = POBJ_ROOT(pop, struct store_root_t);
  assert(!TOID_IS_NULL(root));

  if(D_RO(root)->map.oid.off == 0) {
    PLOG("Root is empty: new hash required");
    //    struct hashmap_args *args = (struct hashmap_args *)arg;
    if(hm_tx_create(pop, &D_RW(root)->map, nullptr))
      throw General_exception("hm_tx_create failed unexpectedly");
  }
  else {
    PLOG("Using existing root:");
    if(hm_tx_init(pop, D_RW(root)->map))
      throw General_exception("hm_tx_init failed unexpectedly");
  }

  if(hm_tx_check(pop, D_RO(root)->map))
    throw General_exception("hm_tx_check failed unexpectedly");

  struct open_session_t * session = new open_session_t;
  session->root = root;
  session->pop = pop;
  session->pool_size = size;
  session->path = fullpath;
  session->io_mem = 0;
  g_sessions.insert(session);

  return reinterpret_cast<uint64_t>(session);
}

void NVME_store::close_pool(pool_t pid)
{
  struct open_session_t * session = reinterpret_cast<struct open_session_t*>(pid);

  if(g_sessions.find(session) == g_sessions.end()){
    PINF("%s: session not here", __func__);
    return;
  }

  g_sessions.erase(session);

  pmemobj_close(session->pop);
  PLOG("NVME_store::closed pool (%lx)", pid);
}

void NVME_store::delete_pool(const pool_t pid)
{
  struct open_session_t * session = reinterpret_cast<struct open_session_t*>(pid);
   
  if(g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store::delete_pool invalid pool identifier");


  g_sessions.erase(session);
  pmemobj_close(session->pop);  
  //TODO should clean the blk_allocator and blk dev (reference) here?
  //_blk_alloc->resize(0, 0);

  if(pmempool_rm(session->path.c_str(), 0))
    throw General_exception("unable to delete pool (%p)", pid);

  PLOG("pool deleted: %s", session->path.c_str());
}

/*
 * when using NVMe, only insert the block range descriptor into the mapping 
 */
status_t NVME_store::put(IKVStore::pool_t pool,
                  std::string key,
                  const void * value,
                  size_t value_len)
{
  struct open_session_t * session = reinterpret_cast<struct open_session_t*>(pool);

  if(g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store::put invalid pool identifier");
  
  auto& root = session->root;
  auto& pop = session->pop;

  auto& blk_alloc = _blk_alloc;
  auto& blk_dev = _blk_dev;

  uint64_t hashkey = CityHash64(key.c_str(), key.length());

  void * handle;

  /* check to see if key already exists */
  if(hm_tx_lookup(pop, D_RO(root)->map, hashkey))
    return E_KEY_EXISTS; 

  size_t nr_io_blocks = (value_len+ BLOCK_SIZE -1)/BLOCK_SIZE;

  // transaction also happens in here
  lba_t lba = blk_alloc->alloc(nr_io_blocks, &handle);

  // TODO: how fast is it?
  io_buffer_t mem = blk_dev->allocate_io_buffer(nr_io_blocks*4096, 4096,Component::NUMA_NODE_ANY);
  PDBG("write to lba %lu with length %lu",lba, value_len);

  TOID(struct block_range) val;
  TX_BEGIN(pop) {

    /* allocate memory for entry - range added to tx implicitly? */
    
    //get the available range from allocator
    val = TX_ALLOC(struct block_range, sizeof(struct block_range));
    
    D_RW(val)->offset = lba;
    D_RW(val)->size = value_len;
    D_RW(val)->handle = handle;

    /* insert into HT */
    int rc;
    if((rc = hm_tx_insert(pop, D_RW(root)->map, hashkey, val.oid))) {
      if(rc == 1)
        return E_ALREADY_EXISTS;
      else throw General_exception("hm_tx_insert failed unexpectedly (rc=%d)", rc);
    }

    memcpy(blk_dev->virt_addr(mem), value, value_len); /* for the moment we have to memcpy */

#ifdef USE_ASYNC
    // TODO: can the free be triggered by callback?
    uint64_t tag = blk_dev->async_write(mem, 0, lba, nr_io_blocks);
    D_RW(val)->last_tag = tag;
#else
    blk_dev->write(mem, 0, lba, nr_io_blocks);
    blk_dev->free_io_buffer(mem);
#endif

  }
  TX_ONABORT {
    //TODO: free val
    throw General_exception("TX abort (%s)", pmemobj_errormsg());
  }
  TX_END

  return S_OK;
}

status_t NVME_store::get(const pool_t pool,
                 const std::string key,
                 void*& out_value,
                 size_t& out_value_len)
{
  struct open_session_t * session = reinterpret_cast<struct open_session_t*>(pool);

  if(g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store::put invalid pool identifier");

  auto& root = session->root;
  auto& pop = session->pop;

  auto& blk_alloc = _blk_alloc;
  auto& blk_dev = _blk_dev;

  uint64_t hashkey = CityHash64(key.c_str(), key.length());

  TOID(struct block_range) val;
  try {
    val = hm_tx_get(pop, D_RW(root)->map, hashkey);
    if(OID_IS_NULL(val.oid))
      return E_KEY_NOT_FOUND;

    auto val_len = D_RO(val)->size;
    auto lba = D_RO(val)->offset;

#ifdef USE_ASYNC
    uint64_t tag = D_RO(val)->last_tag;
    while(!blk_dev->check_completion(tag)) cpu_relax(); /* check the last completion, TODO: check each time makes the get slightly slow () */
#endif

    PDBG("prepare to read lba % lu with length %lu", lba, value_len);
    out_value = malloc(val_len);
    out_value_len = val_len;

    size_t nr_io_blocks = (val_len+ BLOCK_SIZE -1)/BLOCK_SIZE;
    io_buffer_t mem = blk_dev->allocate_io_buffer(nr_io_blocks*4096, 4096,Component::NUMA_NODE_ANY);

    blk_dev->read(mem, 0, lba, nr_io_blocks);

    // transaction also happens in here

    assert(out_value);
    /* memcpy for moment - can i pass the virt_addr(mem) directly? how to free from the client*/
    memcpy(out_value, blk_dev->virt_addr(mem), val_len);

    blk_dev->free_io_buffer(mem);
  }
  catch(...) {
    throw General_exception("hm_tx_get failed unexpectedly");
  }
  return S_OK;
}

status_t NVME_store::get_direct(const pool_t pool,
                              const std::string key,
                              void* out_value,
                              size_t& out_value_len,
                              size_t offset){
  struct open_session_t * session = reinterpret_cast<struct open_session_t*>(pool);

  if(g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store::put invalid pool identifier");

  auto& root = session->root;
  auto& pop = session->pop;

  auto& blk_alloc = _blk_alloc;
  auto& blk_dev = _blk_dev;

  uint64_t hashkey = CityHash64(key.c_str(), key.length());

  TOID(struct block_range) val;
  try {
    val = hm_tx_get(pop, D_RW(root)->map, hashkey);
    if(OID_IS_NULL(val.oid))
      return E_KEY_NOT_FOUND;

    auto val_len = D_RO(val)->size;
    auto lba = D_RO(val)->offset;

#ifdef USE_ASYNC
    uint64_t tag = D_RO(val)->last_tag;
    while(!blk_dev->check_completion(tag)) cpu_relax(); /* check the last completion, TODO: check each time makes the get slightly slow () */
#endif

    PDBG("prepare to read lba % lu with length %lu", lba, value_len);
    assert(out_value);

    /* TODO: safe? */
    io_buffer_t mem = reinterpret_cast<Component::io_buffer_t>(out_value);

    assert(mem);

    size_t nr_io_blocks = (val_len+ BLOCK_SIZE -1)/BLOCK_SIZE;

    blk_dev->read(mem, 0, lba, nr_io_blocks);
    out_value_len = val_len;
  }
  catch(...) {
    throw General_exception("hm_tx_get failed unexpectedly");
  }
  return S_OK;
}


/*
 * Only used for the case when memory is pinned/aligned but not from spdk, e.g. cudadma
 * should be 2MB aligned in both phsycial and virtual*/
status_t NVME_store::register_direct_memory(void * vaddr, size_t len){
  addr_t phys_addr; // physical address
  io_buffer_t handle = 0;;

  phys_addr = xms_get_phys(vaddr);
  handle = _blk_dev->register_memory_for_io(vaddr, phys_addr, len);

  /* save this this registration */
  if(handle){
    PINF("Register vaddr %p with paddr %lu, handle %lu", vaddr, phys_addr, handle );
    return S_OK;
  }
  else{
    PERR("%s: register user allocated memory failed", __func__);
    return E_FAIL;
  }
}



status_t NVME_store::allocate(const pool_t pool,
                      const std::string key,
                      const size_t nbytes,
                      uint64_t& out_key_hash)
{
  open_session_t * session = get_session(pool);
  
  auto& root = session->root;
  auto& pop = session->pop;

  auto& blk_alloc = _blk_alloc;
  auto& blk_dev = _blk_dev;

  uint64_t key_hash = CityHash64(key.c_str(), key.length());

  void * handle;

  /* check to see if key already exists */
  /*if(hm_tx_lookup(pop, d_ro(root)->map, key_hash))*/
    /*return e_key_exists;*/


  size_t nr_io_blocks = (nbytes+ BLOCK_SIZE -1)/BLOCK_SIZE;

  // transaction also happens in here
  lba_t lba = blk_alloc->alloc(nr_io_blocks, &handle);

  TOID(struct block_range) val;

  TX_BEGIN(pop) {

    /* allocate memory for entry - range added to tx implicitly? */
    
    //get the available range from allocator
    val = TX_ALLOC(struct block_range, sizeof(struct block_range));
    
    D_RW(val)->offset = lba;
    D_RW(val)->size = nbytes;
    D_RW(val)->handle = handle;
#ifdef USE_ASYNC
    D_RW(val)->last_tag = 0;
#endif

    /* insert into HT */
    int rc;
    if((rc = hm_tx_insert(pop, D_RW(root)->map, key_hash, val.oid))) {
      if(rc == 1)
        return E_ALREADY_EXISTS;
      else throw General_exception("hm_tx_insert failed unexpectedly (rc=%d)", rc);
    }
  }
  TX_ONABORT {
    //TODO: free blk_range
    throw General_exception("TX abort (%s)", pmemobj_errormsg());
  }
  TX_END

    out_key_hash = key_hash;
  
  return S_OK;
}


/*
 * nvmestore will fetch data from nvme if it obtain the lock
 */
status_t NVME_store::lock(const pool_t pool,
                  uint64_t key_hash,
                  int type,
                  void*& out_value,
                  size_t& out_value_len)
{
  open_session_t * session = get_session(pool);
  
  auto& root = session->root;
  auto& pop = session->pop;

  assert(session->io_mem == 0);
  auto& blk_dev = _blk_dev;

  TOID(struct block_range) val;
  try {
    val = hm_tx_get(pop, D_RW(root)->map, key_hash);
    if(OID_IS_NULL(val.oid))
      return E_KEY_NOT_FOUND;

#ifdef USE_ASYNC
    /* there might be pending async write for this object */
    uint64_t tag = D_RO(val)->last_tag;
    while(!blk_dev->check_completion(tag)) cpu_relax(); /* check the last completion */
#endif

    
    if(type == IKVStore::STORE_LOCK_READ) {
      if(!_sm.state_get_read_lock(pool, D_RO(val)->handle))
        throw General_exception("%s: unable to get read lock", __func__);
    }
    else {
      if(!_sm.state_get_write_lock(pool, D_RO(val)->handle))
        throw General_exception("%s: unable to get write lock", __func__);
    }

    auto handle = D_RO(val)->handle;
    auto value_len = D_RO(val)->size; // the length allocated before
    auto lba = D_RO(val)->offset;

    /* fetch the data to block io mem */
    size_t nr_io_blocks = (value_len + BLOCK_SIZE -1)/BLOCK_SIZE;
    io_buffer_t mem = blk_dev->allocate_io_buffer(nr_io_blocks*4096, 4096,Component::NUMA_NODE_ANY);

    blk_dev->read(mem, 0, lba, nr_io_blocks);
    PINF("NVME_store: read to io memory at %lu", mem);

    session->io_mem = mem; //TODO: can be placed in another place

    /* set output values */
    auto out_value = blk_dev->virt_addr(mem);
    auto out_value_len = value_len;
  }
  catch(...) {
    throw General_exception("hm_tx_get failed unexpectedly");
  }

  PINF("NVME_store: obtained the lock");

  return S_OK;
}


/*
 * this will send async io to nvme and return, the completion will be checked for either get() or the nect lock()/apply
 */
status_t NVME_store::unlock(const pool_t pool,
                    uint64_t key_hash)
{
  open_session_t * session = get_session(pool);
  
  auto& root = session->root;
  auto& pop = session->pop;

  assert(session->io_mem);

  auto& blk_dev = _blk_dev;

  TOID(struct block_range) val;
  try {
    val = hm_tx_get(pop, D_RW(root)->map, key_hash);
    if(OID_IS_NULL(val.oid))
      return E_KEY_NOT_FOUND;

    auto val_len = D_RO(val)->size;
    auto lba = D_RO(val)->offset;

    size_t nr_io_blocks = (val_len + BLOCK_SIZE -1)/BLOCK_SIZE;
    io_buffer_t mem = session->io_mem;

    /*flush and release iomem*/ 
#if USE_ASYNC
    uint64_t tag = blk_dev->async_write(mem, 0, lba, nr_io_blocks);
    D_RW(val)->last_tag = tag;
#else
    blk_dev->write(mem, 0, lba, nr_io_blocks);
    blk_dev->free_io_buffer(mem);
    session->io_mem = 0;
#endif


    PINF("NVME_store: io memory at %lu is freed", mem);

    /*release the lock*/
    _sm.state_unlock(pool, D_RO(val)->handle);

    PINF("NVME_store: released the lock");
  }
  catch(...) {
    throw General_exception("hm_tx_get failed unexpectedly");
  }

  return S_OK;
}

status_t NVME_store::apply(const pool_t pool,
                   const std::string key,
                   std::function<void(void*,const size_t)> functor,
                   size_t offset,
                   size_t size)
{

  void * data;
  size_t value_len = 0;

  lock(pool,CityHash64(key.c_str(), key.length()),IKVStore::STORE_LOCK_READ, data, value_len );
  return __apply(pool,CityHash64(key.c_str(), key.length()),functor, offset, size);
}

status_t NVME_store::apply(const pool_t pool,
                   uint64_t key_hash,
                   std::function<void(void*,const size_t)> functor,
                   size_t offset,
                   size_t size)
{
  void * data;
  size_t value_len = 0;

  lock(pool, key_hash,IKVStore::STORE_LOCK_READ, data, value_len );
  return __apply(pool, key_hash, functor, offset, size);
}

status_t NVME_store::locked_apply(const pool_t pool,
                          const std::string key,
                          std::function<void(void*,const size_t)> functor,
                          size_t offset,
                          size_t size)
{
  return __apply(pool, CityHash64(key.c_str(), key.length()), functor, offset, size);
}

status_t NVME_store::locked_apply(const pool_t pool,
                          uint64_t key_hash,
                          std::function<void(void*,const size_t)> functor,
                          size_t offset,
                          size_t size)
{
  return __apply(pool, key_hash, functor, offset, size);
}

/* currently requires lock from outside
  this will release the lock before returning*/
int NVME_store::__apply(const pool_t pool,
                    uint64_t key_hash,
                    std::function<void(void*,const size_t)> functor,
                    size_t offset,
                    size_t size)
{
  open_session_t * session = get_session(pool);
  
  auto& root = session->root;
  auto& pop = session->pop;

  assert(session->io_mem);
  auto& blk_dev = _blk_dev;

  TOID(struct block_range) val;
  try {
    val = hm_tx_get(pop, D_RW(root)->map, key_hash);
    if(OID_IS_NULL(val.oid))
      return E_KEY_NOT_FOUND;

    auto data = blk_dev->virt_addr(session->io_mem);
    auto data_len = D_RO(val)->size;

    if(offset + size > data_len)
      return E_BAD_PARAM;

    auto offset_data = (void*) (((unsigned long) data) + offset);
    size_t size_to_tx = size > 0 ? size : data_len;

    /* for nvmestore, the change will be synced to nvme when unlocked
     * This can be improved to using pmemobj api*/
    functor(offset_data, size_to_tx); /* execute functor inside of the transaction */
   
#if 0
    TX_BEGIN(pop) {
      pmemobj_tx_add_range_direct(offset_data, size_to_tx); /* add only transaction range */
      functor(offset_data, size_to_tx); /* execute functor inside of the transaction */
    }
    TX_ONABORT {
      throw General_exception("TX abort");
    }
    TX_END
#endif

    unlock(pool, key_hash);
  }
  catch(...) {
    throw General_exception("hm_tx_get failed unexpectedly");
  }

  return S_OK;
}

status_t NVME_store::erase(const pool_t pool,
                   const std::string key)
{
  return erase(pool,CityHash64(key.c_str(), key.length())); 
}

status_t NVME_store::erase(const pool_t pool,
                   uint64_t key_hash)
{
  open_session_t * session = get_session(pool);
  
  auto& root = session->root;
  auto& pop = session->pop;

  auto& blk_alloc = _blk_alloc;
  auto& blk_dev = _blk_dev;

  TOID(struct block_range) val;

  try {
    val = hm_tx_get(pop, D_RW(root)->map, key_hash);
    if(OID_IS_NULL(val.oid))
      return E_KEY_NOT_FOUND;

    /* get hold of write lock to remove */
    if(!_sm.state_get_write_lock(pool, D_RO(val)->handle))
      throw API_exception("unable to remove, value locked");

    blk_alloc->free(D_RO(val)->offset, D_RO(val)->handle);
    
    val = hm_tx_remove(pop, D_RW(root)->map, key_hash); /* could be optimized to not re-lookup */
    if(OID_IS_NULL(val.oid))
      throw API_exception("hm_tx_remove failed unexpectedly");

    _sm.state_remove(pool, D_RO(val)->handle);
  }
  catch(...) {
    throw General_exception("hm_tx_remove failed unexpectedly");
  }
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
