#include <iostream>
#include <set>
#include <errno.h>
#include <libpmemobj.h>
#include <libpmempool.h>
#include <libpmemobj/base.h>

#include <stdio.h>
#include <city.h>
#include <api/kvstore_itf.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/utils.h>
#include <boost/filesystem.hpp>
#include "pm_store.h"

#define USE_TX

#ifdef USE_TX
#define HM_CREATE  hm_tx_create
#define HM_INIT    hm_tx_init
#define HM_CHECK   hm_tx_check
#define HM_CMD     hm_tx_cmd
#define HM_INSERT  hm_tx_insert
#define HM_GET     hm_tx_get
#define HM_REMOVE  hm_tx_remove
#define HM_COUNT   hm_tx_count
#define HM_FOREACH hm_tx_foreach
#define HM_LOOKUP  hm_tx_lookup
#else
#define HM_CREATE  hm_atomic_create
#define HM_INIT    hm_atomic_init
#define HM_CHECK   hm_atomic_check
#define HM_CMD     hm_atomic_cmd
#define HM_INSERT  hm_atomic_insert
#define HM_GET     hm_atomic_get
#define HM_REMOVE  hm_atomic_remove
#define HM_COUNT   hm_atomic_count
#define HM_FOREACH hm_atomic_foreach
#define HM_LOOKUP  hm_atomic_lookup
#endif

extern "C"
{
#include "hashmap_tx.h"
#include "hashmap_atomic.h"
}

#define REGION_NAME "pmstore-default"

using namespace Component;

struct map_value {
  uint64_t len;
  char     data[];
};

struct store_root_t
{
#ifdef USE_TX
  TOID(struct hashmap_tx) map;
#else
  TOID(struct hashmap_atomic) map;
#endif
};
TOID_DECLARE_ROOT(struct store_root_t);

struct open_session_t
{
  TOID(struct store_root_t) root;
  PMEMobjpool *             pop;
  size_t                    pool_size;
  std::string               path;
};

POBJ_LAYOUT_BEGIN(map_value);
POBJ_LAYOUT_TOID(map_value, struct map_value);
POBJ_LAYOUT_TOID(map_value, uint64_t);
POBJ_LAYOUT_END(map_value);

struct tls_cache_t {
  open_session_t * session;
};

// globals
static __thread tls_cache_t tls_cache = { nullptr };
std::set<open_session_t*> g_sessions;

static open_session_t * get_session(IKVStore::pool_t pid) //open_session_t * session)
{
  open_session_t * session = reinterpret_cast<struct open_session_t*>(pid);
  if(session == tls_cache.session) return session;

  if(g_sessions.find(session) == g_sessions.end())
    throw API_exception("PM_store::delete_pool invalid pool identifier");

  return session;
}

static int check_pool(const char * path)
{
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
    return 0;
  }

  return 1;
}


PM_store::PM_store(const std::string& owner, const std::string& name)
{
  PLOG("PMEMOBJ_MAX_ALLOC_SIZE: %lu MB", REDUCE_MB(PMEMOBJ_MAX_ALLOC_SIZE));
}

PM_store::~PM_store()
{
}


IKVStore::pool_t PM_store::create_pool(const std::string& path,
                                       const std::string& name,
                                       const size_t size,
                                       unsigned int flags,
                                       uint64_t args)
{
  PMEMobjpool *pop;

  if(option_DEBUG)
    PLOG("PM_store::create_pool path=%s pool_name=%s", path.c_str(), name.c_str());

  if(size > PMEMOBJ_MAX_ALLOC_SIZE) {
    PWRN("PM_store::create_pool - object too large");
    return E_TOO_LARGE;
  }

  std::string fullpath;
  if(path[path.length()-1]!='/')
    fullpath = path + "/" + name;
  else
    fullpath = path + name;

  if (access(fullpath.c_str(), F_OK) != 0) {
    if(option_DEBUG)
      PLOG("PM_store: creating new pool: %s (%s) size=%lu", name.c_str(), fullpath.c_str(), size);

    boost::filesystem::path p(fullpath);
    boost::filesystem::create_directories(p.parent_path());

    pop = pmemobj_create(fullpath.c_str(), REGION_NAME, size, 0666);
    if(not pop)
      throw General_exception("failed to create new pool - %s\n", pmemobj_errormsg());
  }
  else {
    if(option_DEBUG)
      PLOG("PM_store: opening existing Pool: %s", fullpath.c_str());

    if(check_pool(fullpath.c_str()) != 0)
      throw General_exception("pool check failed");

    pop = pmemobj_open(fullpath.c_str(), REGION_NAME);
    if(not pop)
      throw General_exception("failed to re-open pool - %s\n", pmemobj_errormsg());
  }

  /* see: https://github.com/pmem/pmdk/blob/stable-1.4/src/examples/libpmemobj/map/kv_server.c */

  TOID(struct store_root_t) root = POBJ_ROOT(pop, struct store_root_t);
  assert(!TOID_IS_NULL(root));

  if(D_RO(root)->map.oid.off == 0) {
    //    struct hashmap_args *args = (struct hashmap_args *)arg;
    if(HM_CREATE(pop, &D_RW(root)->map, nullptr))
      throw General_exception("hm_XXX_create failed unexpectedly");
  }
  else {
    if(HM_INIT(pop, D_RW(root)->map))
      throw General_exception("hm_XXX_init failed unexpectedly");
  }

  //  if((flags & FLAGS_SET_SIZE) && (args > 0))
  /* rebuild to pool size */
  //  HM_CMD(pop, D_RO(root)->map, HASHMAP_CMD_REBUILD, size/16);

  if(HM_CHECK(pop, D_RO(root)->map))
    throw General_exception("hm_XXX_check failed unexpectedly (create)");

  struct open_session_t * session = new open_session_t;
  session->root = root;
  session->pop = pop;
  session->pool_size = size;
  session->path = fullpath;
  g_sessions.insert(session);

  return reinterpret_cast<uint64_t>(session);
}

IKVStore::pool_t PM_store::open_pool(const std::string& path,
                                     const std::string& name,
                                     unsigned int flags)
{
  PMEMobjpool *pop = nullptr;

  if (access(path.c_str(), F_OK) != 0)
    throw API_exception("Pool %s:%s does not exist", path.c_str(), name.c_str());

  std::string fullpath = path + name;

  /* check integrity first */
  if(check_pool(fullpath.c_str()) != 0)
    throw General_exception("pool check failed");

  pop = pmemobj_open(fullpath.c_str(), REGION_NAME);
  if(not pop)
    throw General_exception("failed to re-open pool - %s\n", pmemobj_errormsg());

  TOID(struct store_root_t) root = POBJ_ROOT(pop, struct store_root_t);
  if(TOID_IS_NULL(root))
    throw General_exception("Root is NULL!");

  if(D_RO(root)->map.oid.off == 0) {
    if(option_DEBUG)
      PLOG("Root is empty: new hash required");
    //    struct hashmap_args *args = (struct hashmap_args *)arg;
    if(HM_CREATE(pop, &D_RW(root)->map, nullptr))
      throw General_exception("hm_XXX_create failed unexpectedly");
  }
  else {
    if(option_DEBUG)
      PLOG("Using existing root:");
    if(HM_INIT(pop, D_RW(root)->map))
      throw General_exception("hm_XXX_init failed unexpectedly");
  }

  if(HM_CHECK(pop, D_RO(root)->map))
    throw General_exception("hm_XXX_check failed unexpectedly (open)");

  struct open_session_t * session = new open_session_t;
  session->root = root;
  session->pop = pop;
  session->path = fullpath;
  g_sessions.insert(session);

  return reinterpret_cast<uint64_t>(session);
}

void PM_store::close_pool(pool_t pid)
{
  struct open_session_t * session = reinterpret_cast<struct open_session_t*>(pid);

  if(g_sessions.find(session) == g_sessions.end())
    throw API_exception("PM_store::put invalid pool identifier");

  g_sessions.erase(session);

  pmemobj_close(session->pop);
  if(option_DEBUG)
    PLOG("PM_store::closed pool (%lx)", pid);
}

void PM_store::delete_pool(const pool_t pid)
{
  struct open_session_t * session = reinterpret_cast<struct open_session_t*>(pid);

  if(g_sessions.find(session) == g_sessions.end())
    throw API_exception("PM_store::delete_pool invalid pool identifier");

  g_sessions.erase(session);
  pmemobj_close(session->pop);

  if(pmempool_rm(session->path.c_str(), 0)) {
    throw General_exception("unable to delete pool (%s)", session->path.c_str());
  }

  if(option_DEBUG)
    PLOG("pool deleted: %s", session->path.c_str());
}

status_t PM_store::put_direct(const pool_t pool,
                              const std::string& key,
                              const void * value,
                              const size_t value_len,
                              memory_handle_t handle)
{
  /* pm_store can't do DMA yet, so revert to memcpy */
  return put(pool, key, value, value_len);
}

Component::IKVStore::memory_handle_t PM_store::register_direct_memory(void * vaddr, size_t len)
{
  return reinterpret_cast<Component::IKVStore::memory_handle_t>(new iovec{vaddr,len});
}

status_t PM_store::put(IKVStore::pool_t pool,
                       const std::string& key,
                       const void * value,
                       const size_t value_len)
{
  if(option_DEBUG) {
    PLOG("PM_store: put (key=%.*s) (value=%.*s)",
         (int) key.length(), (char*) key.c_str(), (int) value_len, (char*) value);
    assert(value_len > 0);
  }
  
  open_session_t * session = get_session(pool);

  auto& root = session->root;
  const auto& pop = session->pop;

  const uint64_t hashkey = CityHash64(key.c_str(), key.length());

  TOID(struct map_value) val;
  bool exists =  HM_LOOKUP(pop, D_RO(root)->map, hashkey);
  
  TX_BEGIN(pop) {

    if(exists) {
      val = HM_GET(pop, D_RW(root)->map, hashkey);
      if(D_RW(val)->len != value_len)
        throw General_exception("PM_store::put existing object different size");
    }
    else {    
      /* allocate memory for entry - range added to tx implicitly */
      val = TX_ALLOC(struct map_value, sizeof(struct map_value) + value_len);
      /* insert into HT */
      int rc = HM_INSERT(pop, D_RW(root)->map, hashkey, val.oid);
    
      if(rc == 0) {}
      else if(rc == 1)
        throw General_exception("hm_XXX_insert failed already exists");
      else if(rc == -1)
        throw General_exception("hm_XXX_insert failed unexpectedly in Put");
      D_RW(val)->len = value_len;
    }
    
    pmemobj_tx_add_range_direct(D_RO(val)->data, value_len);
    memcpy(D_RW(val)->data, value, value_len); /* for the moment we have to memcpy */

  }
  TX_ONABORT {
    throw General_exception("TX abort (%s)", pmemobj_errormsg());
  }
  TX_END

    return S_OK;
}


status_t PM_store::get(const pool_t pool,
                       const std::string& key,
                       void*& out_value,
                       size_t& out_value_len)
{
  open_session_t * session = get_session(pool);

  auto& root = session->root;
  const auto& pop = session->pop;

  const uint64_t hashkey = CityHash64(key.c_str(), key.length());

  TOID(struct map_value) val;
  try {
    val = HM_GET(pop, D_RO(root)->map, hashkey);
    if(OID_IS_NULL(val.oid)) {
      PWRN("key:%s not found", key.c_str());
      return E_KEY_NOT_FOUND;
    }

    auto val_len = D_RO(val)->len;
    out_value = malloc(val_len);
    out_value_len = val_len;

    assert(out_value);
    /* memcpy for moment - the value can't be moved from underneath because
       of the singleton-thread per pool threading model
    */
    memcpy(out_value, D_RO(val)->data, val_len);
  }
  catch(...) {
    throw General_exception("hm_XXX_get failed unexpectedly");
  }
  return S_OK;
}

status_t PM_store::get_direct(const pool_t pool,
                              const std::string& key,
                              void* out_value,
                              size_t& out_value_len,
                              Component::IKVStore::memory_handle_t handle)
{
  const uint64_t hashkey = CityHash64(key.c_str(), key.length());

  open_session_t * session = get_session(pool);

  auto& root = session->root;
  auto& pop = session->pop;

  TOID(struct map_value) val;
  try {
    val = HM_GET(pop, D_RW(root)->map, hashkey);

    if(OID_IS_NULL(val.oid)) {
      return E_NOT_FOUND;
    }

    auto val_len = D_RO(val)->len;

    if(out_value_len < val_len) {
      PWRN("get_direct failed; insufficient buffer");
      return E_INSUFFICIENT_BUFFER;
    }

    out_value_len = val_len;

    assert(out_value);
    /* memcpy for moment - the value can't be moved from underneath because
       of the singleton-thread per pool threading model
    */
    memcpy(out_value, D_RO(val)->data, val_len);

    if(option_DEBUG)
      PLOG("PM_store: value_len=%lu value=(%s)", val_len, (char*) out_value);
  }
  catch(...) {
    throw General_exception("hm_XXX_get failed unexpectedly");
  }
  return S_OK;
}



Component::IKVStore::key_t
PM_store::lock(const pool_t pool,
               const std::string& key,
               lock_type_t type,
               void*& out_value,
               size_t& out_value_len)
{
  open_session_t * session = get_session(pool);

  auto& root = session->root;
  auto& pop = session->pop;
  auto key_hash = CityHash64(key.c_str(), key.length());
  
  TOID(struct map_value) val;
  TX_BEGIN(pop) {

    val = HM_GET(pop, D_RW(root)->map, key_hash);

    /* if the key is not found, we create it and
       allocate value space equal in size to out_value_len */
    if(OID_IS_NULL(val.oid)) {

      if(out_value_len == 0)
        return Component::IKVStore::KEY_NONE;

      if(option_DEBUG)
        PLOG("PM_store: lock allocating object (%lx) of %lu bytes", key_hash, out_value_len);
      val = TX_ALLOC(struct map_value, sizeof(struct map_value) + out_value_len);
      D_RW(val)->len = out_value_len;

      /* insert into HT */
      int rc;
      if((rc = HM_INSERT(pop, D_RW(root)->map, key_hash, val.oid)))
        throw General_exception("hm_XXX_insert failed unexpectedly (rc=%d)", rc);
    }
    else {
      if(option_DEBUG)
        PLOG("PM_store: lock using existing object (%lx) of %lu bytes", key_hash, D_RO(val)->len);
    }

    auto data = D_RW(val)->data;

    if(type == IKVStore::STORE_LOCK_READ) {
      if(!_sm.state_get_read_lock(pool, data))
        throw General_exception("unable to get read lock");
    }
    else {
      if(!_sm.state_get_write_lock(pool, data))
        throw General_exception("unable to get write lock");
    }

    out_value = data; // (void*) round_up((addr_t)data, 64);
    out_value_len = D_RO(val)->len;
  }
  TX_ONABORT {
    throw General_exception("TX abort in PM_store::lock (%s)", pmemobj_errormsg());
  }
  TX_END

  return reinterpret_cast<Component::IKVStore::key_t>(key_hash);
}


status_t PM_store::unlock(const pool_t pool,
                          Component::IKVStore::key_t key_handle)
{
  open_session_t * session = get_session(pool);

  auto& root = session->root;
  auto& pop = session->pop;

  auto key_hash = reinterpret_cast<uint64_t>(key_handle);

  TOID(struct map_value) val;
  try {
    val = HM_GET(pop, D_RW(root)->map, key_hash);
    if(OID_IS_NULL(val.oid))
      return E_KEY_NOT_FOUND;

    auto data = D_RW(val)->data;

    _sm.state_unlock(pool, data);
  }
  catch(...) {
    throw General_exception("PM_store::unlock - hm_XXX_get failed unexpectedly");
  }

  return S_OK;
}

// uint64_t PM_store::key_hash(const std::string& key)
// {
//   return CityHash64(key.c_str(), key.length());
// }

status_t PM_store::apply(const pool_t pool,
                         const std::string& key,
                         std::function<void(void*,const size_t)> functor,
                         size_t object_size,
                         bool use_lock)
{
  return __apply(pool,CityHash64(key.c_str(), key.length()),functor, object_size, use_lock);
}

status_t PM_store::__apply(const pool_t pool,
                           uint64_t key_hash,
                           std::function<void(void*,const size_t)> functor,
                           size_t object_size,
                           bool take_lock)
{
  open_session_t * session = get_session(pool);

  auto& root = session->root;
  auto& pop = session->pop;

  TOID(struct map_value) val;

  val = HM_GET(pop, D_RW(root)->map, key_hash);

  /* if object not found, create it */
  if(OID_IS_NULL(val.oid)) {

    TX_BEGIN(pop) {
      val = TX_ZALLOC(struct map_value, sizeof(struct map_value) + object_size);

      /* insert into HT */
      int rc;
      if((rc = HM_INSERT(pop, D_RW(root)->map, key_hash, val.oid))) {
        if(rc == 1) return E_ALREADY_EXISTS;
        else throw General_exception("hm_XXX_insert failed unexpectedly (rc=%d)", rc);
      }

      D_RW(val)->len = object_size;
    }
    TX_ONABORT {
      if(pmemobj_tx_errno() == ENOMEM) {
        PWRN("%s out of memory", __PRETTY_FUNCTION__);
        throw General_exception("%s: failed due to out of memory", __PRETTY_FUNCTION__);
      }
      else
        throw General_exception("%s: failed unexpectedly");
    }
    TX_END {}

  }

  auto data = D_RW(val)->data;
  auto data_len = D_RO(val)->len;

  if(take_lock)
    if(!_sm.state_get_write_lock(pool, data))
      throw General_exception("unable to get write lock");

  auto offset_data = (void*) (((unsigned long) data));
  size_t size_to_tx = data_len;

  TX_BEGIN(pop) {
    pmemobj_tx_add_range_direct(offset_data, size_to_tx); /* add only transaction range */
    functor(offset_data, size_to_tx); /* execute functor inside of the transaction */
  }
  TX_ONABORT {
    if(pmemobj_tx_errno() == ENOMEM) {
      PWRN("%s out of memory", __PRETTY_FUNCTION__);
      throw General_exception("%s: failed due to out of memory", __PRETTY_FUNCTION__);
    }
    else
      throw General_exception("%s: failed unexpectedly");
  }
  TX_END {
  }

  if(take_lock)
    _sm.state_unlock(pool, data); /* release lock */

  return S_OK;
}

status_t PM_store::erase(const pool_t pool,
                         const std::string& key)
{
  open_session_t * session = get_session(pool);

  auto& root = session->root;
  auto& pop = session->pop;
  uint64_t key_hash = CityHash64(key.c_str(), key.length());
  TOID(struct map_value) val;
  try {
    val = HM_GET(pop, D_RW(root)->map, key_hash);
    if(OID_IS_NULL(val.oid))
      return E_KEY_NOT_FOUND;

    /* get hold of write lock to remove */
    if(!_sm.state_get_write_lock(pool, D_RO(val)->data))
      throw API_exception("unable to remove, value locked");

    val = HM_REMOVE(pop, D_RW(root)->map, key_hash); /* could be optimized to not re-lookup */
    if(OID_IS_NULL(val.oid))
      throw API_exception("hm_XXX_remove failed unexpectedly");

    _sm.state_remove(pool, D_RO(val)->data);
  }
  catch(...) {
    throw General_exception("hm_XXX_remove failed unexpectedly");
  }
  return S_OK;
}

size_t PM_store::count(const pool_t pool)
{
  open_session_t * session = get_session(pool);

  auto& root = session->root;
  auto& pop = session->pop;

  return HM_COUNT(pop, D_RO(root)->map);
}

void PM_store::debug(const pool_t pool, unsigned cmd, uint64_t arg)
{
  open_session_t * session = get_session(pool);

  auto& root = session->root;
  auto& pop = session->pop;

  HM_CMD(pop, D_RO(root)->map, cmd, arg);
}

static int __functor(uint64_t key, PMEMoid value, void *arg)
{
  assert(arg);
  std::function<int(uint64_t key, const void *val, size_t val_len)> * lambda =
    reinterpret_cast<std::function<int(uint64_t key, const void *val, const size_t val_len)>*>(arg);

  TOID(struct map_value) mv = value;

  (*lambda)(key, D_RO(mv)->data, D_RO(mv)->len);
}

// status_t PM_store::map(pool_t pool,
//                        std::function<int(const std::string& key, const void *val, const size_t val_len)> function)
// {
//   open_session_t * session = get_session(pool);

//   auto& root = session->root;
//   auto& pop = session->pop;

//   if(HM_FOREACH(pop,
//                 D_RO(root)->map,
//                 __functor, (void*) &function)) {
//     throw General_exception("hm_XXX_foreach failed unexpectedly");
//   }

//   return S_OK;
// }

bool PM_store::State_map::state_get_read_lock(const pool_t pool, const void * ptr) {
  state_map_t::accessor a;
  _state_map.insert(a, pool);
  auto& pool_state_map = a->second;
  auto entry = pool_state_map.find(ptr);
  if(entry == pool_state_map.end())  /* create new entry */
    return pool_state_map[ptr]._lock.read_lock() == 0;
  else
    return entry->second._lock.read_trylock() == 0;
}

bool PM_store::State_map::state_get_write_lock(const pool_t pool, const void * ptr) {
  state_map_t::accessor a;
  _state_map.insert(a, pool);
  auto& pool_state_map = a->second;
  auto entry = pool_state_map.find(ptr);
  if(entry == pool_state_map.end())  /* create new entry */
    return pool_state_map[ptr]._lock.write_lock() == 0;
  else
    return entry->second._lock.write_trylock() == 0;
}

void PM_store::State_map::state_unlock(const pool_t pool, const void * ptr) {
  state_map_t::accessor a;
  _state_map.insert(a, pool);
  auto& pool_state_map = a->second;
  auto entry = pool_state_map.find(ptr);
  if(entry == pool_state_map.end() || entry->second._lock.unlock())
    throw General_exception("invalid unlock");
}

void PM_store::State_map::state_remove(const pool_t pool, const void * ptr) {
  state_map_t::accessor a;
  _state_map.insert(a, pool);
  auto& pool_state_map = a->second;
  auto entry = pool_state_map.find(ptr);
  if(entry == pool_state_map.end())
    throw General_exception("invalid remove");
  pool_state_map.erase(entry);
}


/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == PM_store_factory::component_id()) {
    return new PM_store_factory();
  }
  else return NULL;
}

#undef RESET_STATE
