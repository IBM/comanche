/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef PERSIST_SESSION_PMEM_H_
#define PERSIST_SESSION_PMEM_H_

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
#endif

#endif
