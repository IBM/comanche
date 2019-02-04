#ifndef COMANCHE_HSTORE_SESSION_H
#define COMANCHE_HSTORE_SESSION_H

#include "hstore_open_pool.h"

/* open_pool_handle, ALLOC_T, table_t */
template <typename Handle, typename Allocator, typename Table>
  class session
    : public open_pool<Handle>
{
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
      : open_pool<Handle>(path_, std::move(pop_))
#if USE_CC_HEAP == 1
      , _heap(
        Allocator(
          *new
             (pmemobj_direct(heap_oid_))
             Core::cc_alloc(static_cast<char *>(pmemobj_direct(heap_oid_) + sizeof(Core::cc_alloc)))
        )
      )
#elif USE_CC_HEAP == 2
      , _heap(
        Allocator(
          *new
            (pmemobj_direct(heap_oid_))
            Core::heap_co(heap_oid_)
        )
      )
#else /* USE_CC_HEAP */
      , _heap(ALLOC_T(static_cast<open_pool<Handle> *>(this)->pool()))
#endif /* USE_CC_HEAP */
      , _map(persist_data_, _heap)
      , _atomic_state(*persist_data_, _map)
    {}

	explicit session(
		const pool_path &path_
		, Handle &&pop_
	)
		: open_pool<Handle>(path_, std::move(pop_))
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
    typename Table::key_type &key
    , std::vector<Component::IKVStore::Operation *>::const_iterator first
    , std::vector<Component::IKVStore::Operation *>::const_iterator last
  ) -> Component::status_t
  {
    return _atomic_state.enter(allocator(), key, first, last);
  }
};

#endif
