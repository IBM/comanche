#ifndef COMANCHE_HSTORE_NUPM2_H
#define COMANCHE_HSTORE_NUPM2_H

/* requires persist_data_t definition */

class region
{
public:
  static constexpr std::uint64_t magic_value = 0xc74892d72eed493a;
  std::uint64_t magic;
  persist_data_t persist_data;
  Core::cc_alloc heap;
  /* region used by cc_alloc follows */
};

namespace
{
  void map_create(
    region *pop_
    , std::size_t size_
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

    auto *p = &pop_->persist_data;
    PLOG(PREFIX "created persist_data ptr at addr %p", __func__, static_cast<const void *>(p));
    void *a = &pop_->heap;
    auto actual_size = size_ - sizeof(region);
    PLOG(PREFIX "created heap at addr %pi region size %zu heap size %zu", __func__, static_cast<const void *>(a), size_, actual_size);
    /* arguments to cc_malloc are the start of the free space (which cc_sbrk uses
     * for the "state" structure) and the size of the free space
     */
    auto al = new (a) Core::cc_alloc(static_cast<char *>(a) + sizeof(Core::cc_alloc), actual_size);
    new (p) persist_data_t(
      expected_obj_count
      , table_t::allocator_type(*al)
    );
    pop_->magic = region::magic_value;
    Persister::persist(pop_, sizeof *pop_);
  }

  void map_create_if_null(
                         region *pop_
                         , std::size_t size_
                         , std::size_t expected_obj_count
                         , bool verbose
                         )
  {
    const bool initialized = pop_->magic == region::magic_value;
    if ( ! initialized )
    {
      map_create(pop_, size_, expected_obj_count, verbose);
    }
  }
}

/* requires ALLOC_T */

class session
  : public open_pool
{
  ALLOC_T                   _heap;
  table_t                   _map;
  impl::atomic_controller<table_t> _atomic_state;
public:
  explicit session(
                        const std::string &dir_
                        , const std::string &name_
                        , open_pool_handle &&pop_
                        )
    : open_pool(dir_, name_, std::move(pop_))
    , _heap(
		ALLOC_T(
			*new
				(&pop_->heap)
				Core::cc_alloc(pop_.get() + 1)
		)
	)
    , _map(&pop_->persist_data, _heap)
    , _atomic_state(pop_->persist_data, _map)
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

auto Nupm_create_pool(
  std::shared_ptr<nupm::Devdax_manager> dax_mgr_
  , unsigned numa_node_
  , const std::string &dir_
  , const std::string &name_
  , std::size_t size_
  , std::size_t expected_obj_count_
  , bool option_DEBUG_
  ) -> std::unique_ptr<session>
{
  open_pool_handle pop = create_or_open_pool(dax_mgr_, numa_node_, dir_, name_, size_, option_DEBUG_);
  map_create_if_null(pop.get(), size_, expected_obj_count_, option_DEBUG_);
  auto s = std::make_unique<session>(dir_, name_, std::move(pop));
  return s;
}

auto Nupm_open_pool(
  std::shared_ptr<nupm::Devdax_manager> dax_mgr_
  , unsigned numa_node_
  , const std::string &dir_
  , const std::string &name_) -> std::unique_ptr<::open_pool>
{
  const std::string path = make_full_path(dir_, name_);
  auto uuid = dax_uuid_hash(path.c_str());
  if (
      auto pop =
        open_pool_handle(static_cast<region *>(dax_mgr_->open_region(uuid, numa_node_, nullptr)), region_closer(dax_mgr_))
      )
  {
    /* open_pool returns either a ::open_pool (usable for delete_pool) or a ::session
     * (usable for delete_pool and everything else), depending on whether the region
     * data is usuable for all operations or just for deletion.
     */
    try
    {
      return std::make_unique<session>(dir_, name_, std::move(pop));
    }
    catch ( ... )
    {
      return std::make_unique<::open_pool>(dir_, name_, std::move(pop));
    }
  }
  else
  {
    auto e = errno;
    throw General_exception("failed to re-open region %s: %s", path.c_str(), std::strerror(e));
  }
}

#endif
