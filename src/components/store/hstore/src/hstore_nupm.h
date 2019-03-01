/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef COMANCHE_HSTORE_NUPM_H
#define COMANCHE_HSTORE_NUPM_H

#include "hstore_pm.h"

#include "hstore_common.h"
#include "persister_nupm.h"
#include "dax_map.h"

#include <cstring> /* strerror */

#include "region.h"

#include "hstore_session.h"

#include <cinttypes> /* PRIx64 */
#include <cstdlib> /* getenv */

using Persister = persister_nupm;

namespace
{
  unsigned name_to_numa_node(const std::string &name)
  {
    if ( 0 == name.size() )
    {
      throw std::domain_error("cannot determine numa node from null string");
    }
    auto c = name[name.size()-1];
    if ( ! std::isprint(c) )
    {
      throw std::domain_error("last character of name (unprintable) does not look like a numa node ID");
    }
    if ( c < '0' || '8' < c )
    {
#if 0
      throw std::domain_error(std::string("last character of name '") + name + "' does not look like a numa node ID");
#else
      /* current test cases do not always supply a node number - default to 0 */
      c = '0';
#endif
    }
    return c - '0';
  }
}

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
  std::unique_ptr<Devdax_manager> _devdax_manager;
  unsigned _numa_node;

  static std::uint64_t dax_uuid_hash(const pool_path &p)
  {
    std::string s = p.str();
    return CityHash64(s.data(), s.size());
  }

  void *delete_and_recreate_pool(const pool_path &path_, std::size_t size_)
  {
    auto uuid = dax_uuid_hash(path_);
    _devdax_manager->erase_region(uuid, _numa_node);

    auto pop = _devdax_manager->create_region(uuid, _numa_node, size_);
    if (not pop) {
      auto e = errno;
      throw std::system_error(std::error_code(e, std::system_category()), std::string("failed to create region ") + path_.str());
    }
    PLOG(PREFIX "in %s: created region ID %" PRIx64 " at %p:0x%zx", __func__, path_.str().c_str(), uuid, pop, size_);
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
#if USE_CC_HEAP == 3
    auto al = new (a) heap_rc(static_cast<char *>(a) + sizeof(heap_rc), actual_size, _numa_node);
#else
    auto al = new (a) heap_cc(static_cast<char *>(a) + sizeof(heap_cc), actual_size);
#endif
    new (p) persist_data_t(
      expected_obj_count
      , table_t::allocator_type(*al)
    );
    pop_->initialize();
    Persister::persist(pop_, sizeof *pop_);
  }

  bool debug() { return false; }
public:
  hstore_nupm(const std::string &, const std::string &name_, std::unique_ptr<Devdax_manager> mgr_, bool debug_)
    : pool_manager(debug_)
    , _devdax_manager(std::move(mgr_))
    , _numa_node(name_to_numa_node(name_))
  {}

  virtual ~hstore_nupm() {}

  auto pool_create_check(std::size_t) -> status_t override
  {
    return S_OK;
  }

  auto pool_create(
    const pool_path &path_
    , std::size_t size_
    , int flags_
    , std::size_t expected_obj_count_
  ) -> std::unique_ptr<tracked_pool> override
  {
    if ( flags_ != 0 )
    {
      throw pool_error("unsupported flags " + std::to_string(flags_), pool_ec::pool_unsupported_mode);
    }
    auto uuid = dax_uuid_hash(path_);
    /* Attempt to create a new pool. */
    auto pop = open_pool_handle(static_cast<region *>(_devdax_manager->create_region(uuid, _numa_node, size_)), region_closer(shared_from_this()));
    /* Guess that nullptr indicate a failure */
    if ( ! pop )
    {
      throw pool_error("create_region fail: " + path_.str(), pool_ec::region_fail);
    }
    PLOG(PREFIX "in %s: created region ID %" PRIx64 " at %p:0x%zx", __func__, path_.str().c_str(), uuid, static_cast<const void *>(pop.get()), size_);

    map_create(pop.get(), size_, expected_obj_count_);
    return std::make_unique<session<open_pool_handle, ALLOC_T, table_t>>(path_, std::move(pop), construction_mode::create);
  }

  auto pool_open(
    const pool_path &path_
    , int flags_
  ) -> std::unique_ptr<tracked_pool> override
  {
    if ( flags_ != 0 )
    {
      throw pool_error("unsupported flags " + std::to_string(flags_), pool_ec::pool_unsupported_mode);
    }
    auto uuid = dax_uuid_hash(path_);
    auto pop = open_pool_handle(static_cast<region *>(_devdax_manager->open_region(uuid, _numa_node, nullptr)), region_closer(shared_from_this()));
    if ( ! pop )
    {
      throw std::invalid_argument("failed to re-open pool");
    }

    void *a = &pop->heap;

#if USE_CC_HEAP == 3
    /* reconstituted heap */
    new (a) heap_rc(static_cast<char *>(a) + sizeof(heap_rc));
#else
    new (a) heap_cc(static_cast<char *>(a) + sizeof(heap_cc));
#endif

    PLOG(PREFIX "in %s: opened region ID %" PRIx64 " at %p", __func__, path_.str().c_str(), uuid, static_cast<const void *>(pop.get()));
    /* open_pool_handle is a managed region *, and pop is a region. */
    auto s = std::make_unique<session<open_pool_handle, ALLOC_T, table_t>>(path_, std::move(pop), construction_mode::reconstitute);
    return s;
  }

  void pool_close_check(const std::string &) override
  {
  }

  void pool_delete(const pool_path &path_) override
  {
    auto uuid = dax_uuid_hash(path_);
    _devdax_manager->erase_region(uuid, _numa_node);
  }

  /* ERROR: want get_pool_regions(<proper type>, std::vector<::iovec>&) */
  status_t pool_get_regions(void *, std::vector<::iovec>&) override
  {
    return E_NOT_SUPPORTED;
  }
};
#pragma GCC diagnostic pop

#endif
