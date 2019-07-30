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

#undef USE_PMEM  // Deprecated.
#ifdef USE_PMEM
#include "persist_session_pmem.h"
#endif

using namespace Component;

namespace fs=boost::filesystem;

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

NVME_store::NVME_store(const std::string& owner,
                       const std::string& name,
                       const std::string& pci,
                       const std::string& pm_path,
                       persist_type_t     persist_type)
    : _pm_path(pm_path), _metastore(owner, name, pm_path, persist_type),
      _blk_manager(pci, pm_path, _metastore)
{
  // order: pm_path -> metastore -> block allocator(might use metastore)

  // path
  if (_pm_path.back() != '/') _pm_path += "/";
}

NVME_store::~NVME_store() {}

IKVStore::pool_t NVME_store::create_pool(const std::string& name,
                                         const size_t       size,
                                         unsigned int       flags,
                                         uint64_t           args)
{
  int ret = 0;

  // TODO: need to check size

#if USE_PMEM
  const std::string& fullpath = _pm_path + name;

  PINF("[NVME_store]::create_pool fullpath=%s name=%s", fullpath.c_str(),
       name.c_str());

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
      _metastore.get_store()->create_pool(name, estimated_obj_map_size, flags);

  // Creating pool with same name
  if (obj_info_pool == POOL_ERROR) {
    PWRN("[%s:%d]:Creating objmap pool failed with name %s, size = %ld, flags = 0x, pool already exists?%x", __FUNCTION__, __LINE__,
        name.c_str(), estimated_obj_map_size, flags);
    return POOL_ERROR;
  }
  open_session_t* session =
      new open_session_t(_metastore.get_store(), obj_info_pool, name,
                         DEFAULT_IO_MEM_SIZE, &_blk_manager, &_sm);

  g_sessions.insert(session);

  return reinterpret_cast<uint64_t>(session);
}

IKVStore::pool_t NVME_store::open_pool(const std::string& name,
                                       unsigned int       flags)
{

#ifdef USE_PMEM
  const std::string& fullpath = _pm_path + name;
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
  IKVStore::pool_t obj_info_pool =
      _metastore.get_store()->open_pool(name, flags);
  if (obj_info_pool == POOL_ERROR) {
    throw General_exception("objmap pool: failed during opening ");
  }

  open_session_t* session =
      new open_session_t(_metastore.get_store(), obj_info_pool, name,
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
  if (S_OK !=
      _metastore.get_store()->close_pool(session->get_obj_info_pool())) {
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

  /* if trying to open a unclosed pool!*/
  for (auto iter : g_sessions) {
    if (iter->get_path() == name) {
      PWRN("nvmestore: try to delete an opened pool!");
      return E_ALREADY_OPEN;
    }
  }
#ifdef USE_PMEM
  const std::string& fullpath = _pm_path + name;
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
  if (S_OK != _metastore.get_store()->delete_pool(name)) {
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

status_t NVME_store::put_direct(IKVStore::pool_t   pool,
                         const std::string& key,
                         const void*        value,
                         size_t             value_len,
                         memory_handle_t memory_handle,
                         unsigned int       flags)
{
  NVME_store::put(pool, key, value, value_len, flags);
}

status_t NVME_store::get(const pool_t       pool,
                         const std::string& key,
                         void*&             out_value,
                         size_t&            out_value_len)
{
  open_session_t* session = reinterpret_cast<open_session_t*>(pool);

  if (g_sessions.find(session) == g_sessions.end())
    throw API_exception("NVME_store::get invalid pool identifier");

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
    throw API_exception("NVME_store::get_direct invalid pool identifier");

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
status_t NVME_store::allocate_direct_memory(
    void*&                     vaddr,
    size_t                     len,
    IKVStore::memory_handle_t& out_handle)
{
  io_buffer_t io_mem;
  size_t      blk_sz = _blk_manager.blk_sz();
  io_mem = _blk_manager.allocate_io_buffer(round_up(len, blk_sz), 4096,
                                           Component::NUMA_NODE_ANY);
  if (io_mem == 0)
    throw API_exception("NVME_store:: direct memory allocation failed");
  vaddr = _blk_manager.virt_addr(io_mem);

  buffer_t* buffer = new buffer_t(len, io_mem, vaddr);

  out_handle = reinterpret_cast<IKVStore::memory_handle_t>(buffer);
  /* save this this registration */
  return S_OK;
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
status_t NVME_store::lock(const pool_t       pool,
                          const std::string& key,
                          lock_type_t        type,
                          void*&             out_value,
                          size_t&            out_value_len,
                          IKVStore::key_t&   out_key)
{
  open_session_t* session = get_session(pool);

  session->lock(key, type, out_value, out_value_len);
  PDBG("[nvmestore_lock] %p", out_value);
  out_key = reinterpret_cast<Component::IKVStore::key_t>(out_value);
  return S_OK;
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

  persist_type_t meta_persist_type = PERSIST_FILE;
  if (params.find("persist_type") != params.end()) {
    if (params["persist_type"] == "filestore") {
      meta_persist_type = PERSIST_FILE;
    }
    else if (params["persist_type"] == "hstore") {
      meta_persist_type = PERSIST_HSTORE;
      if ( fs::exists("/dev/dax0.1") ||
          (getenv("USE_DRAM") == NULL)) {
        throw General_exception(
            "[nvmestore factory]: Need to set in current shell: \n \texport "
            "USE_DRAM=24; export NO_CLFLUSHOPT=1; export DAX_RESET=1");
      }
    }
    else {
      throw API_exception("Option %s not supported",
                          params["persist_type"].c_str());
    }
  }

  Component::IKVStore* obj = static_cast<Component::IKVStore*>(
      new NVME_store(params["owner"], params["name"], params["pci"],
                     params["pm_path"], meta_persist_type));
  obj->add_ref();
  return obj;
}
