#ifndef COMANCHE_HSTORE_NUPM_H
#define COMANCHE_HSTORE_NUPM_H

#define USE_CC_HEAP 1

#include "hstore_common.h"
#include "persister_nupm.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#pragma GCC diagnostic ignored "-Weffc++"
#include <nupm/dax_map.h>
#pragma GCC diagnostic pop

#include <cstring> /* strerror */

class region;

class region_closer
{
	std::shared_ptr<nupm::Devdax_manager> _mgr;
public:
	region_closer(std::shared_ptr<nupm::Devdax_manager> mgr_)
		: _mgr(mgr_)
	{}
	void operator()(region *) noexcept
	{
#if 0
		/* Note: There is not yet a way to close a region.  And when there is,
		 * the name may be close_region rather than region_close.
		 */
		_mgr->region_close(r);
#endif
	}
};

using open_pool_handle = std::unique_ptr<region, region_closer>;

using Persister = persister_nupm;

namespace
{
  static std::uint64_t dax_uuid_hash(const std::string &s)
  {
    return CityHash64(s.data(), s.size());
  }
}

namespace
{
  void *delete_and_recreate_pool(std::shared_ptr<nupm::Devdax_manager> mgr_, int numa_node_, const std::string &path_, std::size_t size_)
  {
    auto uuid = dax_uuid_hash(path_);
    mgr_->erase_region(uuid, numa_node_);

    auto pop = mgr_->create_region(uuid, numa_node_, size_);
    if (not pop) {
      auto e = errno;
      throw General_exception("failed to create region (%s) %s", path_, std::strerror(e));
    }
    return pop;
  }
}

class open_pool
{
  std::string               _dir;
  std::string               _name;
  open_pool_handle          _pop;
public:
  explicit open_pool(
    const std::string &dir_
    , const std::string &name_
    , open_pool_handle &&pop_
  )
    : _dir(dir_)
    , _name(name_)
    , _pop(std::move(pop_))
  {}
  open_pool(const open_pool &) = delete;
  open_pool& operator=(const open_pool &) = delete;
  virtual ~open_pool() {}

#if 1
  /* get_pool_regions only */
  auto *pool() const { return _pop.get(); }
#endif
#if 1
  /* delete_pool only */
  const std::string &dir() const noexcept { return _dir; }
  const std::string &name() const noexcept { return _name; }
#endif
};

auto Nupm_make_devdax_manager() -> std::shared_ptr<nupm::Devdax_manager> { return std::make_shared<nupm::Devdax_manager>(); }

namespace
{
  open_pool_handle create_or_open_pool(
    std::shared_ptr<nupm::Devdax_manager> mgr_
    , int numa_node_
    , const std::string &dir_
    , const std::string &name_
    , std::size_t size_
    , bool option_DEBUG_
  )
  {
    auto path = make_full_path(dir_, name_);

    auto uuid = dax_uuid_hash(path);
    /* First, attempt to create a new pool. If that fails, open an existing pool. */
    try
    {
      auto pop = open_pool_handle(static_cast<region *>(mgr_->create_region(uuid, numa_node_, size_)), region_closer(mgr_));
      /* Guess that nullptr indicate a failure */
      if ( ! pop )
      {
        throw std::runtime_error("Failed to create region " + path);
      }
      return pop;
    }
    catch ( const std::exception & )
    {
      if ( option_DEBUG_ )
      {
        PLOG(PREFIX "opening existing Pool: %s", __func__, path.c_str());
      }
      std::size_t existing_size = 0;
      auto pop = open_pool_handle(static_cast<region *>(mgr_->open_region(uuid, numa_node_, &existing_size)), region_closer(mgr_));
      /* assume that size mismatch is reason enough to destroy and recreate the pool */
      if ( existing_size != size_ )
      {
        pop.reset(static_cast<region *>(delete_and_recreate_pool(mgr_, numa_node_, path, size_)));
      }
      return pop;
    }
  }
}

void Nupm_close_pool_check_pool(const std::string &)
{
}

void Nupm_delete_pool(nupm::Devdax_manager &dax_mgr_, unsigned numa_node_, const std::string &path)
{
  auto uuid = dax_uuid_hash(path.c_str());
  dax_mgr_.erase_region(uuid, numa_node_);
}

status_t Nupm_get_pool_regions(region *, std::vector<::iovec>&)
{
  return E_NOT_SUPPORTED;
}

#endif
