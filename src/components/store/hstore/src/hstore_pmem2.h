#ifndef COMANCHE_HSTORE_PMEM2_H
#define COMANCHE_HSTORE_PMEM2_H

namespace
{
  struct root_anchors
  {
    persist_data_t *persist_data_ptr;
    void *heap_ptr;
  };

  auto Pmem_map_open(TOID(struct store_root_t) &root) -> root_anchors
  {
    auto rt = read_const_root(root);
    PLOG(PREFIX "persist root addr %p", __func__, static_cast<const void *>(rt));
    auto apc = pmemobj_direct(rt->persist_oid);
    auto heap = pmemobj_direct(rt->heap_oid);
    PLOG(PREFIX "persist data addr %p", __func__, static_cast<const void *>(apc));
    PLOG(PREFIX "persist heap addr %p", __func__, static_cast<const void *>(heap));
    return root_anchors{static_cast<persist_data_t *>(apc), static_cast<void *>(heap)};
  }

  void Pmem_map_create(
    PMEMobjpool *pop_
    , TOID(struct store_root_t) &root
    , std::size_t
#if USE_CC_HEAP == 1 || USE_CC_HEAP == 2
        size_
#endif /* USE_CC_HEAP */
    , std::size_t expected_obj_count
    , bool verbose
    )
  {
    if ( verbose )
    {
      PLOG(
           PREFIX "root is empty: new hash required object count %zu"
           , __func__
           , expected_obj_count
           );
    }
  auto persist_oid =
    palloc(
           pop_
           , sizeof(persist_data_t)
           , type_num::persist
           , "persist"
           );
    auto *p = static_cast<persist_data_t *>(pmemobj_direct(persist_oid));
    PLOG(PREFIX "created persist_data ptr at addr %p", __func__, static_cast<const void *>(p));

#if USE_CC_HEAP == 1
  auto heap_oid_and_size =
    palloc(
           pop_
           , 64U /* least acceptable size */
           , size_ /* preferred size */
           , type_num::heap
           , "heap"
           );

    auto heap_oid = std::get<0>(heap_oid_and_size);
    auto *a = static_cast<void *>(pmemobj_direct(heap_oid));
    auto actual_size = std::get<1>(heap_oid_and_size);
    PLOG(PREFIX "created heap at addr %p preferred size %zu actual size %zu", __func__, static_cast<const void *>(a), size_, actual_size);
    /* arguments to cc_malloc are the start of the free space (which cc_sbrk uses
     * for the "state" structure) and the size of the free space
     */
    auto al = new (a) Core::cc_alloc(static_cast<char *>(a) + sizeof(Core::cc_alloc), actual_size - sizeof(Core::cc_alloc));
    new (p) persist_data_t(
      expected_obj_count
      , table_t::allocator_type(*al)
    );
    Persister::persist(p, sizeof *p);
#elif USE_CC_HEAP == 2
  auto heap_oid_and_size =
    palloc(
           pop_
           , 64U /* least acceptable size */
           , size_ /* preferred size */
           , type_num::heap
           , "heap"
           );

    auto heap_oid = std::get<0>(heap_oid_and_size);
    auto *a = static_cast<void *>(pmemobj_direct(heap_oid));
    auto actual_size = std::get<1>(heap_oid_and_size);
    PLOG(PREFIX "createed heap at addr %p preferred size %zu actual size %zu", __func__, static_cast<const void *>(a), size_, actual_size);
    /* arguments to cc_malloc are the start of the free space (which cc_sbrk uses
     * for the "state" structure) and the size of the free space
     */
    auto al = new (a) Core::heap_co(heap_oid, actual_size, sizeof(Core::heap_co));
    new (p) persist_data_t(
      expected_obj_count
      , table_t::allocator_type(*al)
    );
    Persister::persist(p, sizeof *p);
#else /* USE_CC_HEAP */
    new (p) persist_data_t(expected_obj_count, table_t::allocator_type{pop_});
    table_t::allocator_type{pop_}
      .persist(p, sizeof *p, "persist_data");
#endif /* USE_CC_HEAP */

#if USE_CC_HEAP == 1
    read_root(root)->heap_oid = heap_oid;
#elif USE_CC_HEAP == 2
    read_root(root)->heap_oid = heap_oid;
#else /* USE_CC_HEAP */
#endif /* USE_CC_HEAP */
    read_root(root)->persist_oid = persist_oid;
  }

  auto Pmem_map_create_if_null(
                         PMEMobjpool *pop_
                         , TOID(struct store_root_t) &root
                         , std::size_t size_
                         , std::size_t expected_obj_count
                         , bool verbose
                         ) -> root_anchors
  {
    const bool initialized = ! OID_IS_NULL(read_const_root(root)->persist_oid);
    if ( ! initialized )
    {
      Pmem_map_create(pop_, root, size_, expected_obj_count, verbose);
    }
    return Pmem_map_open(root);
  }
}

class session
  : public open_pool
{
  ALLOC_T                   _heap;
  table_t                   _map;
  impl::atomic_controller<table_t> _atomic_state;
public:
  explicit session(
                        TOID(struct store_root_t) &
#if USE_CC_HEAP == 1 || USE_CC_HEAP == 2
                         root_
#endif
                        , const std::string &dir_
                        , const std::string &name_
                        , open_pool_handle &&pop_
                        , persist_data_t *persist_data_
                        )
    : open_pool(dir_, name_, std::move(pop_))
#if USE_CC_HEAP == 1
    , _heap(
		ALLOC_T(
			*new
				(pmemobj_direct(read_const_root(root_)->heap_oid))
				Core::cc_alloc(static_cast<char *>(pmemobj_direct(read_const_root(root_)->heap_oid)) + sizeof(Core::cc_alloc))
		)
	)
#elif USE_CC_HEAP == 2
    , _heap(
		ALLOC_T(
			*new
				(pmemobj_direct(read_const_root(root_)->heap_oid))
				Core::heap_co(read_const_root(root_)->heap_oid)
		)
	)
#else /* USE_CC_HEAP */
    , _heap(ALLOC_T(pool()))
#endif /* USE_CC_HEAP */
    , _map(persist_data_, _heap)
    , _atomic_state(*persist_data_, _map)
  {}

  session(const session &) = delete;
  session& operator=(const session &) = delete;
  auto allocator() const { return _heap; }
  table_t &map() noexcept { return _map; }
  const table_t &map() const noexcept { return _map; }

  auto enter(
             KEY_T &key
             , std::vector<Component::IKVStore::Operation *>::const_iterator first
             , std::vector<Component::IKVStore::Operation *>::const_iterator last
             ) -> Component::status_t
  {
    return _atomic_state.enter(allocator(), key, first, last);
  }
};

auto Pmem_create_pool(const std::string &dir_, const std::string &name_, std::size_t size_, std::size_t expected_obj_count, bool option_DEBUG_) -> std::unique_ptr<session>
{
  std::string fullpath = make_full_path(dir_, name_);
  open_pool_handle pop = create_or_open_pool(dir_, name_, size_, option_DEBUG_);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  TOID(struct store_root_t) root = POBJ_ROOT(pop.get(), struct store_root_t);
#pragma GCC diagnostic pop
  assert(!TOID_IS_NULL(root));

  auto pc =
    Pmem_map_create_if_null(
      pop.get(), root, size_, expected_obj_count, option_DEBUG_
    );
  return std::make_unique<session>(root, dir_, name_, std::move(pop), pc.persist_data_ptr);
}

auto Pmem_open_pool(const std::string &dir_
                       , const std::string &name_
                       ) -> std::unique_ptr<::open_pool>
{
  std::string path = make_full_path(dir_, name_);
  if (access(dir_.c_str(), F_OK) != 0)
  {
    throw API_exception("Pool %s:%s does not exist", dir_.c_str(), name_.c_str());
  }

  /* check integrity first */
  if (check_pool(path.c_str()) != 0)
  {
    throw General_exception("pool check failed");
  }

  if (
      auto pop =
        open_pool_handle(pmemobj_open_guarded(path.c_str(), REGION_NAME), pmemobj_close_guarded)
      )
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
    TOID(struct store_root_t) root = POBJ_ROOT(pop.get(), struct store_root_t);
#pragma GCC diagnostic pop
    if (TOID_IS_NULL(root))
    {
      throw General_exception("Root is NULL!");
    }

    auto anchors = Pmem_map_open(root);

    if ( ! anchors.persist_data_ptr )
    {
      throw General_exception("failed to re-open pool (not initialized)");
    }

    /* open_pool returns either a ::open_pool (usable for delete_pool) or a ::session
     * (usable for delete_pool and everything else), depending on whether the pool
     * data is usuable for all operations or just for deletion.
     */
    try
    {
      return std::make_unique<session>(root, dir_, name_, std::move(pop), anchors.persist_data_ptr);
    }
    catch ( ... )
    {
      return std::make_unique<::open_pool>(dir_, name_, std::move(pop));
    }
  }
  else
  {
    throw General_exception("failed to re-open pool %s - %s", path.c_str(), pmemobj_errormsg());
  }
}

#endif
