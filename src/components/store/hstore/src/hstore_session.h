/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef COMANCHE_HSTORE_SESSION_H
#define COMANCHE_HSTORE_SESSION_H

#include "hstore_tracked_pool.h"

#include "construction_mode.h"

#include <utility> /* move */
#include <vector>

/* open_pool_handle, ALLOC_T, table_t */
template <typename Handle, typename Allocator, typename Table>
	class session
		: public tracked_pool
{
	Handle _pop;
	Allocator _heap;
	Table _map;
	impl::atomic_controller<Table> _atomic_state;
public:
	/* PMEMoid, persist_data_t */
	template <typename OID, typename Persist>
		explicit session(
			OID
#if USE_CC_HEAP == 1 || USE_CC_HEAP == 2
				heap_oid_
#endif
			, const pool_path &path_
			, open_pool_handle &&pop_
			, Persist *persist_data_
		)
		: tracked_pool(path_)
		, _pop(std::move(pop_))
		, _heap(
			Allocator(
#if USE_CC_HEAP == 0
				this->pool()
#elif USE_CC_HEAP == 1
				*new
					(pmemobj_direct(heap_oid_))
					heap_cc(static_cast<char *>(pmemobj_direct(heap_oid_) + sizeof(heap_cc)))
#elif USE_CC_HEAP == 2
				*new
					(pmemobj_direct(heap_oid_))
					heap_co(heap_oid_)
#else /* USE_CC_HEAP */
				this->pool() /* not used */
#endif /* USE_CC_HEAP */
			)
		)
		, _map(persist_data_, _heap)
		, _atomic_state(*persist_data_, _map)
	{}

	explicit session(
		const pool_path &path_
		, Handle &&pop_
		, construction_mode mode_
	)
		: tracked_pool(path_)
		, _pop(std::move(pop_))
		, _heap(
			Allocator(
				this->pool()->heap
			)
		)
		, _map(&this->pool()->persist_data, mode_, _heap)
		, _atomic_state(this->pool()->persist_data, _map, mode_)
	{}

  session(const session &) = delete;
  session& operator=(const session &) = delete;
  /* session constructor and get_pool_regions only */
  auto *pool() const { return _pop.get(); }
  auto allocator() const { return _heap; }
  table_t &map() noexcept { return _map; }
  const table_t &map() const noexcept { return _map; }

  auto enter_update(
    typename Table::key_type &key
    , std::vector<Component::IKVStore::Operation *>::const_iterator first
    , std::vector<Component::IKVStore::Operation *>::const_iterator last
  ) -> Component::status_t
  {
    return _atomic_state.enter_update(allocator(), key, first, last);
  }
  auto enter_replace(
    typename Table::key_type &key
    , const void *data
    , std::size_t data_len
  ) -> Component::status_t
  {
    return _atomic_state.enter_replace(allocator(), key, static_cast<const char *>(data), data_len);
  }
};

#endif
