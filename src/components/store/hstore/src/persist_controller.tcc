/*
 * Hopscotch hash table - template Key, Value, and allocators
 */

/*
 * ===== persist_control =====
 */

/*
 * persist testing:
 * 1. Every write to *_persist should be atomic: not more than 8 bytes, and
 *    ideally to an atomic type. In practice, integral and pointer writes are
 *    atomic.
 * 2. perishable::tick() immediately before every write to *__persist allows
 *    crash inject.
 */

template <typename Allocator>
	impl::persist_controller<Allocator>::persist_controller(
		const Allocator &av_
		, persist_data_t *persist_
		, construction_mode mode_
	)
		: Allocator(av_)
		, _persist(persist_)
		, _bucket_count_cached(bucket_count_uncached())
	{
		assert(_persist->_segment_count._target <= _segment_capacity);
		assert(1U <= _persist->_segment_count._target);
		assert(
			_persist->_segment_count._actual <= _persist->_segment_count._target
		);

		/* Persisted data needs at least one segment. */
		if ( _persist->_segment_count._actual == 0 )
		{
			_persist->do_initial_allocation(av_);
		}
		if ( mode_ == construction_mode::reconstitute )
		{
			_persist->reconstitute(av_);
		}
	}

template <typename Allocator>
	void impl::persist_controller<Allocator>::persist_segment_count()
	{
		persist_internal(
			&_persist->_segment_count
			, &_persist->_segment_count+1U
			, "count"
		);
	}

template <typename Allocator>
	void impl::persist_controller<Allocator>::persist_owner(
		const owner &c_
		, const char *why_
	)
	{
		persist_internal(&c_, &c_ + 1U, why_);
	}

template <typename Allocator>
	void impl::persist_controller<Allocator>::persist_content(
		const content_t &c_
		, const char *why_
	)
	{
		/* content is offset from start of cache line;
		 * hash_bucket is cache-aligned. Assuming that
		 * a cache-aligned persist will not be worse, specify it.
		 */
		auto &hb = static_cast<const hash_bucket<value_type> &>(c_);
		/* bucket_aligned_t is the full size of the cache line.
		 * Assuming that full cache line-sized persist will not
		 * be worse, specify it.
		 */
		auto &ba = static_cast<const bucket_aligned_t &>(hb);
		persist_internal(&ba, &ba + 1U, why_);
	}

template <typename Allocator>
	void impl::persist_controller<Allocator>::persist_internal(
		const void *first_
		, const void *last_
		, const char *
#if 0
			what_
#endif
	)
	{
		this->Allocator::persist(first_, static_cast<const char *>(last_) - static_cast<const char *>(first_));
	}

template <typename Allocator>
	void impl::persist_controller<Allocator>::persist_existing_segments(const char *)
	{
/* This persist goes through pmemobj address translation to find the addresses.
 * That is unecessary, as the virtual addresses are kept (in a separate table).
 */
		auto sc = &*_persist->_sc;
		{
			auto bp = &*sc[0].bp;
			persist_internal(&bp[0], &bp[base_segment_size], "segment 0");
		}
		for ( auto i = 1U; i != segment_count_actual(); ++i )
		{
			auto bp = &*sc[i].bp;
			persist_internal(&bp[0], &bp[base_segment_size<<(i-1U)], "segment N");
		}
	}

template <typename Allocator>
	void impl::persist_controller<Allocator>::persist_new_segment(const char *)
	{
/* This persist goes through pmemobj address translation to find the addresses.
 * That is unnecessary, as the virtual addresses are kept (in a separate table).
 */
		auto sc = &*_persist->_sc;
		auto bp = &*sc[segment_count_actual()].bp;
		persist_internal(
			&bp[0]
			, &bp[base_segment_size<<(segment_count_actual()-1U)]
			, "segment new"
		);
	}

template <typename Allocator>
	void impl::persist_controller<Allocator>::persist_segment_table()
	{
/* This persist goes through pmemobj address translation to find the addresses.
 * That su unnecessary, as the virtual addresses are kept (in a separate table).
 */
		auto sc = &*_persist->_sc;
		persist_internal(&sc[0], &sc[persist_data_t::_segment_capacity], "segments");
	}

template <typename Allocator>
	void impl::persist_controller<Allocator>::persist_size()
	{
		persist_internal(
			&_persist->_size_control
			, (&_persist->_size_control)+1U
			, "size"
		);
	}

template <typename Allocator>
	bool impl::persist_controller<Allocator>::is_size_unstable() const
	{
		return ! _persist->_size_control.is_stable();
	}

template <typename Allocator>
	void impl::persist_controller<Allocator>::size_set(std::size_t n)
	{
		_persist->_size_control.size_set_stable(n);
		persist_size();
	}

template <typename Allocator>
	void impl::persist_controller<Allocator>::size_destabilize()
	{
		_persist->_size_control.destabilize();
		persist_size();
	}

template <typename Allocator>
	void impl::persist_controller<Allocator>::size_stabilize()
	{
		_persist->_size_control.stabilize();
		persist_size();
	}

template <typename Allocator>
	auto impl::persist_controller<Allocator>::resize_prolog(
	) -> bucket_aligned<hash_bucket<value_type>> * /* bucket_aligned_t */
	{
		_persist->_segment_count._target = _persist->_segment_count._actual + 1U;
		persist_segment_count();

		using void_allocator_t =
			typename bucket_allocator_t::template rebind<void>::other;

		auto ptr =
			bucket_allocator_t(*this).allocate(
				bucket_count()
				, typename void_allocator_t::const_pointer()
				, "resize"
			);
		new (&*ptr) typename persist_data_t::bucket_aligned_t[bucket_count()];
		_persist->_sc[_persist->_segment_count._actual].bp = ptr;

		persist_segment_table();

		auto sc = &*_persist->_sc;
		return &*(sc[segment_count_actual()].bp);
	}

template <typename Allocator>
	void impl::persist_controller<Allocator>::resize_epilog()
	{
		_persist->_segment_count._actual = _persist->_segment_count._target;
		_bucket_count_cached = bucket_count_uncached();
		persist_segment_count();
	}
