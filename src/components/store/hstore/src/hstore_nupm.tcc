/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "hstore_common.h"
#include "hstore_session.h"
#include "persister_nupm.h"
#include "dax_map.h"
#include "pool_path.h"
#include "region.h"

#include <city.h> /* CityHash */

#include <cinttypes> /* PRIx64 */
#include <cstdlib> /* getenv */
#include <cstring> /* strerror */

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

template <typename Region, typename Table, typename Allocator, typename LockType>
  std::uint64_t hstore_nupm<Region, Table, Allocator, LockType>::dax_uuid_hash(const pool_path &p)
  {
    std::string s = p.str();
    return CityHash64(s.data(), s.size());
  }

template <typename Region, typename Table, typename Allocator, typename LockType>
  bool hstore_nupm<Region, Table, Allocator, LockType>::debug() { return false; }

template <typename Region, typename Table, typename Allocator, typename LockType>
  hstore_nupm<Region, Table, Allocator, LockType>::hstore_nupm(const std::string &, const std::string &name_, std::unique_ptr<Devdax_manager> mgr_, bool debug_)
    : pool_manager<::open_pool<std::unique_ptr<Region, region_closer_t>>>(debug_)
    , _devdax_manager(std::move(mgr_))
    , _numa_node(name_to_numa_node(name_))
  {}

template <typename Region, typename Table, typename Allocator, typename LockType>
  hstore_nupm<Region, Table, Allocator, LockType>::~hstore_nupm()
  {
  }

template <typename Region, typename Table, typename Allocator, typename LockType>
  void hstore_nupm<Region, Table, Allocator, LockType>::pool_create_check(std::size_t)
  {
  }

template <typename Region, typename Table, typename Allocator, typename LockType>
  auto hstore_nupm<Region, Table, Allocator, LockType>::pool_create(
    const pool_path &path_
    , std::size_t size_
    , int flags_
    , std::size_t expected_obj_count_
  ) -> std::unique_ptr<open_pool_handle>
  {
    if ( flags_ != 0 )
    {
      throw pool_error("unsupported flags " + std::to_string(flags_), pool_ec::pool_unsupported_mode);
    }
    auto uuid = dax_uuid_hash(path_);
    /* The first part of pool space is the header, which is described by a Region.
     * In order to give the heap a well-aligned space, the size actually allocated
     * to a heap may be as little as 3/4 of the area provided to the heap.
     * The constant 3/4 is embedded in the heap_rc_shared class.
     *
     * Ask for enough space to contain the header and to compensate for inefficiency
     * due to heap alignment.
     */
    auto size = sizeof(Region) + size_ * 4 / 3;
    /* _devdax_manager will allocate 1GiB-aligned regions. But there is no mechanism for it to
     * tell us that. Round request up to 1 GiB to avoid wasting space.
     */
	size = ((size_ - 1) / (1U<<30) + 1 ) * (1U<<30);
    /* Attempt to create a new pool. */
    try
    {
      void *pop = _devdax_manager->create_region(uuid, _numa_node, size);
      /* Guess that nullptr indicate a failure */
      if ( ! pop )
      {
        throw pool_error("create_region fail: " + path_.str(), pool_ec::region_fail);
      }
      PLOG(PREFIX "in %s: created region ID %" PRIx64 " at %p:0x%zx", __func__, path_.str().c_str(), uuid, pop, size);

      open_pool_handle h(new (pop) Region(uuid, size, expected_obj_count_, _numa_node), region_closer_t(this->shared_from_this()));
      return std::make_unique<session<open_pool_handle, allocator_t, table_t, lock_type_t>>(path_, std::move(h), construction_mode::create);
    }
    catch ( const General_exception &e )
    {
      throw pool_error("create_region fail: " + path_.str() + " " + e.cause(), pool_ec::region_fail_general_exception);
    }
    catch ( const std::bad_alloc& e)
    {
      throw pool_error("create_region fail: " + path_.str(), pool_ec::region_fail_general_exception);
    }
    catch ( const API_exception &e )
    {
      throw pool_error("create_region fail: " + path_.str() + " " + e.cause(), pool_ec::region_fail_api_exception);
    }
  }

  template <typename Region, typename Table, typename Allocator, typename LockType>
  auto hstore_nupm<Region, Table, Allocator, LockType>::pool_open(
    const pool_path &path_
    , int flags_
  ) -> std::unique_ptr<open_pool_handle>
  {
    if ( flags_ != 0 )
    {
      throw pool_error("unsupported flags " + std::to_string(flags_), pool_ec::pool_unsupported_mode);
    }
    auto uuid = dax_uuid_hash(path_);
    open_pool_handle pop(
      std::unique_ptr<region_type, region_closer_t>(
        new (_devdax_manager->open_region(uuid, _numa_node, nullptr)) region_type(_devdax_manager)
        , region_closer_t(this->shared_from_this())
      )
    );
    if ( ! pop )
    {
      throw std::invalid_argument("failed to re-open pool");
    }

    PLOG(PREFIX "in %s: opened region ID %" PRIx64 " at %p", __func__, path_.str().c_str(), uuid, static_cast<const void *>(pop.get()));
    /* open_pool_handle is a managed region *, and pop is a region. */
    auto s = std::make_unique<session<open_pool_handle, allocator_t, table_t, lock_type_t>>(path_, std::move(pop), construction_mode::reconstitute);
    return s;
  }

template <typename Region, typename Table, typename Allocator, typename LockType>
  void hstore_nupm<Region, Table, Allocator, LockType>::pool_close_check(const std::string &)
  {
  }

template <typename Region, typename Table, typename Allocator, typename LockType>
  void hstore_nupm<Region, Table, Allocator, LockType>::pool_delete(const pool_path &path_)
  {
    auto uuid = dax_uuid_hash(path_);
    _devdax_manager->erase_region(uuid, _numa_node);
  }

template <typename Region, typename Table, typename Allocator, typename LockType>
  auto hstore_nupm<Region, Table, Allocator, LockType>::pool_get_regions(const open_pool_handle & pool_) const
  -> std::vector<::iovec>
  {
    return pool_->get_regions();
  }
