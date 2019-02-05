#ifndef COMANCHE_HSTORE_NUPM_H
#define COMANCHE_HSTORE_NUPM_H

#include "hstore_pm.h"

#include "hstore_common.h"
#include "persister_nupm.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#pragma GCC diagnostic ignored "-Weffc++"
#include <nupm/dax_map.h>
#pragma GCC diagnostic pop

#include <cstring> /* strerror */

#include "region.h"

#include "hstore_open_pool.h"

#include "hstore_session.h"

#include <cinttypes> /* PRIx64 */
#include <cstdlib> /* getenv */

using Persister = persister_nupm;

/* open_pool_handle, ALLOC_T, table_t */
template <typename Handle, typename Allocator, typename Table>
  class session;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
class hstore_nupm
  : public pool_manager
  , public std::enable_shared_from_this<hstore_nupm>
{
public:
  using open_pool_handle = std::unique_ptr<region, region_closer>;
private:
  nupm::Devdax_manager _devdax_manager;
  unsigned _numa_node;

  static std::uint64_t dax_uuid_hash(const pool_path &p)
  {
    std::string s = p.str();
    return CityHash64(s.data(), s.size());
  }

  void *delete_and_recreate_pool(const pool_path &path_, std::size_t size_)
  {
    auto uuid = dax_uuid_hash(path_);
    _devdax_manager.erase_region(uuid, _numa_node);

    auto pop = _devdax_manager.create_region(uuid, _numa_node, size_);
    if (not pop) {
      auto e = errno;
      throw General_exception("failed to create region (%s) %s", path_, std::strerror(e));
    }
    PLOG(PREFIX "in %s: created region ID %" PRIx64 " at %p:0x%zx", __func__, path_.str().c_str(), uuid, pop, size_);
    return pop;
  }
#if 0
  open_pool_handle pool_create_or_open(
    const pool_path &path_
    , std::size_t size_
  )
  {
    auto uuid = dax_uuid_hash(path_);
    /* First, attempt to create a new pool. If that fails, open an existing pool. */
    try
    {
      auto pop = open_pool_handle(static_cast<region *>(_devdax_manager.create_region(uuid, _numa_node, size_)), region_closer(shared_from_this()));
      /* Guess that nullptr indicate a failure */
      if ( ! pop )
      {
        throw std::runtime_error("Failed to create region " + path_.str());
      }
      PLOG(PREFIX "in %s: created region ID %" PRIx64 " at %p:0x%zx", __func__, path_.str().c_str(), uuid, static_cast<const void *>(pop.get()), size_);
      return pop;
    }
    catch ( const std::exception & )
    {
      if ( debug() )
      {
        PLOG(PREFIX "opening existing Pool: %s", __func__, path_.str().c_str());
      }
      std::size_t existing_size = 0;
      auto pop = open_pool_handle(static_cast<region *>(_devdax_manager.open_region(uuid, _numa_node, &existing_size)), region_closer(shared_from_this()));
/* Disabled "existing size" check as open_region does not yet return an existing size */
#if 0
      /* assume that size mismatch is reason enough to destroy and recreate the pool */
      if ( existing_size != size_ )
      {
        pop.reset(static_cast<region *>(delete_and_recreate_pool(path_, size_)));
      }
#endif
      PLOG(PREFIX "in %s: opened region ID %" PRIx64 " at %p", __func__, path_.str().c_str(), uuid, static_cast<const void *>(pop.get()));
      return pop;
    }
  }
#endif
  /* Ordinarily we would try create first, but dax map manager allows us to create duplicates. to avoid that,
   * try open first.
   */
  open_pool_handle pool_open_or_create(
    const pool_path &path_
    , std::size_t size_
  )
  {
    auto uuid = dax_uuid_hash(path_);
    /* First, attempt to open an existing pool. If that fails, create a new pool. */
    if ( debug() )
    {
      PLOG(PREFIX "opening existing Pool: %s", __func__, path_.str().c_str());
    }
    std::size_t existing_size = 0;
    auto pop = open_pool_handle(static_cast<region *>(_devdax_manager.open_region(uuid, _numa_node, &existing_size)), region_closer(shared_from_this()));
/* Disabled "existing size" check as open_region does not yet return an existing size */
#if 0
      /* assume that size mismatch is reason enough to destroy and recreate the pool */
      if ( existing_size != size_ )
      {
        pop.reset(static_cast<region *>(delete_and_recreate_pool(path_, size_)));
      }
#endif

    if ( pop )
    {
      PLOG(PREFIX "in %s: opened region ID %" PRIx64 " at %p", __func__, path_.str().c_str(), uuid, static_cast<const void *>(pop.get()));
    }
    else
    {
      pop = open_pool_handle(static_cast<region *>(_devdax_manager.create_region(uuid, _numa_node, size_)), region_closer(shared_from_this()));
      /* Guess that nullptr indicate a failure */
      if ( ! pop )
      {
        throw std::runtime_error("Failed to create region " + path_.str());
      }
      PLOG(PREFIX "in %s: created region ID %" PRIx64 " at %p:0x%zx", __func__, path_.str().c_str(), uuid, static_cast<const void *>(pop.get()), size_);
    }
    return pop;
  }

  void map_create(
    region *pop_
    , std::size_t size_
    , std::size_t expected_obj_count
  )
  {
    if ( debug() )
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
    PLOG(PREFIX "created heap at addr %p region size 0x%zx heap size 0x%zx", __func__, static_cast<const void *>(a), size_, actual_size);
    /* arguments to cc_malloc are the start of the free space (which cc_sbrk uses
     * for the "state" structure) and the size of the free space
     */
    auto al = new (a) Core::cc_alloc(static_cast<char *>(a) + sizeof(Core::cc_alloc), actual_size);
    new (p) persist_data_t(
      expected_obj_count
      , table_t::allocator_type(*al)
    );
    pop_->initialize();
    Persister::persist(pop_, sizeof *pop_);
  }

  void map_create_if_null(
    region *pop_
    , std::size_t size_
    , std::size_t expected_obj_count
  )
  {
    if ( ! pop_->is_initialized() )
    {
      map_create(pop_, size_, expected_obj_count);
    }
  }
  bool debug() { return false; }
public:
  hstore_nupm(unsigned numa_mode_, bool debug_)
    : pool_manager(debug_)
    , _devdax_manager{bool(std::getenv("DAX_RESET"))}
    , _numa_node(numa_mode_)
  {}

  virtual ~hstore_nupm() {}

  auto pool_create_check(std::size_t) -> status_t override
  {
    return S_OK;
  }

  auto pool_create(
    const pool_path &path_
    , std::size_t size_
    , std::size_t expected_obj_count_
  ) -> std::unique_ptr<tracked_pool> override
  {
    open_pool_handle pop = pool_open_or_create(path_, size_);
    map_create_if_null(pop.get(), size_, expected_obj_count_);
    return std::make_unique<session<open_pool_handle, ALLOC_T, table_t>>(path_, std::move(pop));
  }

  auto pool_open(
    const pool_path &path_) -> std::unique_ptr<tracked_pool> override
  {
    auto uuid = dax_uuid_hash(path_);
    auto pop = open_pool_handle(static_cast<region *>(_devdax_manager.open_region(uuid, _numa_node, nullptr)), region_closer(shared_from_this()));
    if ( ! pop )
    {
      auto e = errno;
      throw General_exception("failed to re-open region %s: %s", path_.str().c_str(), std::strerror(e));
    }
    PLOG(PREFIX "in %s: opened region ID %" PRIx64 " at %p", __func__, path_.str().c_str(), uuid, static_cast<const void *>(pop.get()));
    /* open_pool returns either a ::open_pool (usable for delete_pool) or a ::session
     * (usable for delete_pool and everything else), depending on whether the region
     * data is usuable for all operations or just for deletion.
     */
    try
    {
      return std::make_unique<session<open_pool_handle, ALLOC_T, table_t>>(path_, std::move(pop));
    }
    catch ( ... )
    {
      return std::make_unique<open_pool<open_pool_handle>>(path_, std::move(pop));
    }
  }

  void pool_close_check(const std::string &) override
  {
  }

  void pool_delete(const pool_path &path_) override
  {
    auto uuid = dax_uuid_hash(path_);
    _devdax_manager.erase_region(uuid, _numa_node);
  }

  /* ERROR: want get_pool_regions(<proper type>, std::vector<::iovec>&) */
  status_t pool_get_regions(void *, std::vector<::iovec>&) override
  {
    return E_NOT_SUPPORTED;
  }
};
#pragma GCC diagnostic pop

#endif
