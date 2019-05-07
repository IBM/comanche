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


#ifndef COMANCHE_HSTORE_HEAP_RC_H
#define COMANCHE_HSTORE_HEAP_RC_H

#include "dax_map.h"
#include "histogram_log2.h"
#include "hop_hash_log.h"
#include "persister_nupm.h"
#include "rc_alloc_wrapper_lb.h"
#include "trace_flags.h"

#include <boost/icl/interval_set.hpp>
#if 0
#include <valgrind/memcheck.h>
#else
#define VALGRIND_CREATE_MEMPOOL(pool, x, y) do {} while(0)
#define VALGRIND_DESTROY_MEMPOOL(pool) do {} while(0)
#define VALGRIND_MAKE_MEM_DEFINED(pool, size) do {} while(0)
#define VALGRIND_MAKE_MEM_UNDEFINED(pool, size) do {} while(0)
#define VALGRIND_MEMPOOL_ALLOC(pool, addr, size) do {} while(0)
#define VALGRIND_MEMPOOL_FREE(pool, size) do {} while(0)
#endif
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <common/exceptions.h> /* General_exception */
#pragma GCC diagnostic pop

#include <sys/uio.h> /* iovec */

#include <algorithm>
#include <cassert>
#include <cstddef> /* size_t, ptrdiff_t */
#include <memory>
#include <new> /* std::bad_alloc */

class heap_rc_shared_ephemeral
{
	nupm::Rca_LB _heap;
	std::size_t _allocated;
	std::size_t _used;
	std::size_t _capacity;
	using alloc_set_t = boost::icl::interval_set<const char *>; /* std::byte_t in C++17 */
	alloc_set_t _reconstituted; /* std::byte_t in C++17 */
	using hist_type = util::histogram_log2<std::size_t>;
	hist_type _hist_alloc;
	hist_type _hist_inject;
	hist_type _hist_free;

	static constexpr unsigned log_alignment = 6U;
	static constexpr unsigned hist_report_upper_bound = 34U;
	template <bool B>
		void write_hist(const ::iovec & pool_) const
		{
			static bool suppress = false;
			if ( ! suppress )
			{
				hop_hash_log<B>::write(__func__, " pool ", pool_.iov_base);
				std::size_t lower_bound = 0;
				for ( unsigned i = std::max(0U, log_alignment); i != std::min(std::size_t(hist_report_upper_bound), _hist_alloc.data().size()); ++i )
				{
					const std::size_t upper_bound = 1ULL << i;
					hop_hash_log<B>::write(__func__, " [", lower_bound, "..", upper_bound, "): ", _hist_alloc.data()[i], " ", _hist_inject.data()[i], " ", _hist_free.data()[i], " ");
					lower_bound = upper_bound;
				}
				suppress = true;
			}
		}
public:
	friend class heap_rc_shared;

	heap_rc_shared_ephemeral(std::size_t capacity_)
		: _heap()
		, _allocated(0)
		, _used(0)
		, _capacity(capacity_)
		, _reconstituted()
		, _hist_alloc()
		, _hist_inject()
		, _hist_free()
	{}
};

class heap_rc_shared
{
	static constexpr std::size_t alignment = 64U;
	::iovec _pool0;
	unsigned _numa_node;
	std::unique_ptr<heap_rc_shared_ephemeral> _eph;
	std::size_t _more_region_uuids_size;
	std::array<std::uint64_t, 1024U> _more_region_uuids;
	/* Rca_LB seems not to allocate at or above about 2GiB. Limit reporting to 16 GiB. */
	/* The set of reconstituted addresses. Only needed during recovery.
	 * Potentially large, so should be erased after recovery. But there
	 * is no mechanism to erase it yet.
	 */
	static void *best_aligned(void *a, std::size_t sz_)
	{
		const auto begin = reinterpret_cast<uintptr_t>(a);
		const auto end = begin + sz_;
		auto cursor = begin + sz_ - 1U;

		/* find best-aligned address in [begin, end)
		 * by removing ones from largest possible address
		 * until further removal would precede begin.
		 */
		{
			auto next_cursor = cursor & (cursor - 1U);
			while ( begin <= next_cursor )
			{
				cursor = next_cursor;
				next_cursor &= (next_cursor - 1U);
			}
		}

		auto best_alignemnt = cursor;
		/* Best alignment, but maybe too small. Need to move toward begin to reduce lost space. */
		/* isolate low one bit */
		{
			auto bit = ( cursor & - cursor ) >> 1;

			/* arbitrary size requirement: 3/4 of the availble space */
			/* while ( (end - cursor) / sz_ < 3/4 ) */
			while ( (end - cursor) < sz_ * 3/4 )
			{
				auto next_cursor = cursor - bit;
				if ( begin <= next_cursor )
				{
					cursor = next_cursor;
				}
				bit >>= 1;
			}
		}
		hop_hash_log<TRACE_HEAP_SUMMARY>::write(
			__func__, " range [", std::hex, begin, "..", end, ")",
			" best aligned ", std::hex, best_alignemnt, " 3/4-space at ", std::hex, cursor
		);

		return reinterpret_cast<void *>(cursor);
	}
	static ::iovec align(void *pool_, std::size_t sz_)
	{
		auto pool = best_aligned(pool_, sz_);
		return
			::iovec{
				pool
				, std::size_t((static_cast<char *>(pool_) + sz_) - static_cast<char *>(pool))
			};
	}
public:
	heap_rc_shared(void *pool_, std::size_t sz_, unsigned numa_node_)
		: _pool0(align(pool_, sz_))
		, _numa_node(numa_node_)
		, _eph(std::make_unique<heap_rc_shared_ephemeral>(_pool0.iov_len))
		, _more_region_uuids_size(0)
		, _more_region_uuids()
	{
		/* cursor now locates the best-aligned region */
		_eph->_heap.add_managed_region(_pool0.iov_base, _pool0.iov_len, _numa_node);
		hop_hash_log<TRACE_HEAP_SUMMARY>::write(
			__func__, " this ", this
			, " pool ", _pool0.iov_base, " .. ", iov_limit(_pool0)
			, " size ", _pool0.iov_len
			, " new"
		);
		VALGRIND_CREATE_MEMPOOL(_pool0.iov_base, 0, false);
	}

	~heap_rc_shared()
	{
		quiesce();
	}

	static ::iovec open_region(const std::unique_ptr<Devdax_manager> &devdax_manager_, std::uint64_t uuid_, unsigned numa_node_)
	{
		::iovec iov;
		iov.iov_base = devdax_manager_->open_region(uuid_, numa_node_, &iov.iov_len);
		if ( iov.iov_base == 0 )
		{
			throw std::range_error("failed to re-open region " + std::to_string(uuid_));
		}
		return iov;
	}

	void animate(const std::unique_ptr<Devdax_manager> &devdax_manager_)
	{
		_eph = std::make_unique<heap_rc_shared_ephemeral>(_pool0.iov_len);
		_eph->_heap.add_managed_region(_pool0.iov_base, _pool0.iov_len, _numa_node);
		hop_hash_log<TRACE_HEAP_SUMMARY>::write(
			__func__, " this ", this
			, " pool ", _pool0.iov_base, " .. ", iov_limit(_pool0)
			, " size ", _pool0.iov_len
			, " reconstituting"
		);
		VALGRIND_MAKE_MEM_DEFINED(_pool0.iov_base, _pool0.iov_len);
		VALGRIND_CREATE_MEMPOOL(_pool0.iov_base, 0, true);
		for ( std::size_t i = 0; i != _more_region_uuids_size; ++i )
		{
#if 0
			_eph._more_regions.push_back(open_region(devdax_manager_, _more_region_uuids[i], _numa_node));
			auto &r = _eph._more_regions.back();
#else
			auto r = open_region(devdax_manager_, _more_region_uuids[i], _numa_node);
#endif
			_eph->_heap.add_managed_region(r.iov_base, r.iov_len, _numa_node);
			_eph->_capacity += r.iov_len;
			VALGRIND_MAKE_MEM_DEFINED(r.iov_base, r.iov_len);
			VALGRIND_CREATE_MEMPOOL(r.iov_base, 0, true);
		}
	}

	static void *iov_limit(const ::iovec &r)
	{
		return static_cast<char *>(r.iov_base) + r.iov_len;
	}

	auto grow(
		const std::unique_ptr<Devdax_manager> & devdax_manager_
		, std::uint64_t uuid_
		, std::size_t increment_
	) -> std::size_t
	{
		if ( 0 < increment_ )
		{
			if ( _more_region_uuids_size == _more_region_uuids.size() )
			{
				throw std::bad_alloc(); /* max # of regions used */
			}
			auto size = ( (increment_ - 1) / (1U<<30) + 1 ) * (1U<<30);
			auto uuid = _more_region_uuids_size == 0 ? uuid_ : _more_region_uuids[_more_region_uuids_size-1];
			auto uuid_next = uuid + 1;
			for ( ; uuid_next != uuid; ++uuid_next )
			{
				if ( uuid_next != 0 )
				{
					try
					{
						/* Note: crash between here and "Slot persist done" may cause devdax_manager_
						 * to leak the region.
						 */
						::iovec r { devdax_manager_->create_region(uuid_next, _numa_node, size), size };
						{
							auto &slot = _more_region_uuids[_more_region_uuids_size];
							slot = uuid_next;
							persister_nupm::persist(&slot, sizeof slot);
							/* Slot persist done */
						}
						{
							++_more_region_uuids_size;
							persister_nupm::persist(&_more_region_uuids_size, _more_region_uuids_size);
						}
						_eph->_heap.add_managed_region(r.iov_base, r.iov_len, _numa_node);
						_eph->_capacity += size;
						hop_hash_log<TRACE_HEAP_SUMMARY>::write(
							__func__, " this ", this
							, " pool ", r.iov_base, " .. ", iov_limit(r)
							, " size ", r.iov_len
							, " grow"
						);
						break;
					}
					catch ( const std::bad_alloc & )
					{
						/* probably means that the uuid is in use */
					}
					catch ( const General_exception & )
					{
						/* probably means that the space cannot be allocated */
						throw std::bad_alloc();
					}
				}
			}
			if ( uuid_next == uuid )
			{
				throw std::bad_alloc(); /* no more UUIDs */
			}
		}
		return _eph->_capacity;
	}

	void quiesce()
	{
		hop_hash_log<TRACE_HEAP_SUMMARY>::write(__func__, " this ", this, " size ", _pool0.iov_len, " allocated ", _eph->_allocated, " used ", _eph->_used);
		VALGRIND_DESTROY_MEMPOOL(_pool0.iov_base);
		VALGRIND_MAKE_MEM_UNDEFINED(_pool0.iov_base, _pool0.iov_len);
		_eph->write_hist<TRACE_HEAP_SUMMARY>(_pool0);
		_eph.reset(nullptr);
	}

	heap_rc_shared(const heap_rc_shared &) = delete;
	heap_rc_shared &operator=(const heap_rc_shared &) = delete;

	void *alloc(std::size_t sz_)
	{
		/* allocation must be multiple of alignment */
		auto sz = (sz_ + alignment - 1U)/alignment * alignment;

		try {
			auto p = _eph->_heap.alloc(sz, _numa_node, alignment);
					/* Note: allocation exception from Rca_LB is General_exception, which does not derive
					 * from std::bad_alloc.
					 */

			VALGRIND_MEMPOOL_ALLOC(_pool0.iov_base, p, sz);
			hop_hash_log<TRACE_HEAP>::write(__func__, " pool ", _pool0.iov_base, " addr ", p, " size ", sz_, " -> ", sz);
			_eph->_used += sz_;
			_eph->_allocated += sz;
			_eph->_hist_alloc.enter(sz);
			return p;
		}
		catch ( const std::bad_alloc & )
		{
			_eph->write_hist<true>(_pool0);
			/* Sometimes lack of space will cause heap to throw a bad_alloc. */
			throw;
		}
		catch ( const General_exception &e )
		{
			_eph->write_hist<true>(_pool0);
			/* Sometimes lack of space will cause heap to throw a General_exception with this explanation. */
			/* Convert to bad_alloc. */
			if ( e.cause() == std::string("region allocation out-of-space") )
			{
				throw std::bad_alloc();
			}
			throw;
		}
	}

	void inject_allocation(const void * p, std::size_t sz_)
	{
		auto sz = (sz_ + alignment - 1U)/alignment * alignment;
		/* NOTE: inject_allocation should take a const void* */
		_eph->_heap.inject_allocation(const_cast<void *>(p), sz, _numa_node);
		VALGRIND_MEMPOOL_ALLOC(_pool0.iov_base, p, sz);
		hop_hash_log<TRACE_HEAP>::write(__func__, " pool ", _pool0.iov_base, " addr ", p, " size ", sz);

		{
			auto pc = static_cast<heap_rc_shared_ephemeral::alloc_set_t::element_type>(p);
			_eph->_reconstituted.add(heap_rc_shared_ephemeral::alloc_set_t::segment_type(pc, pc + sz));
		}
		_eph->_used += sz_;
		_eph->_allocated += sz;
		_eph->_hist_inject.enter(sz);
	}

	void free(void *p_, std::size_t sz_)
	{
		auto sz = (sz_ + alignment - 1U)/alignment * alignment;
		VALGRIND_MEMPOOL_FREE(_pool0.iov_base, p_);
		hop_hash_log<TRACE_HEAP>::write(__func__, " pool ", _pool0.iov_base, " addr ", p_, " size ", sz);
		_eph->_used -= sz_;
		_eph->_allocated -= sz;
		_eph->_hist_free.enter(sz);
		return _eph->_heap.free(p_, _numa_node, sz);
	}

	bool is_reconstituted(const void * p_) const
	{
		return contains(_eph->_reconstituted, static_cast<heap_rc_shared_ephemeral::alloc_set_t::element_type>(p_));
	}

	/* debug */
	unsigned numa_node() const
	{
		return _numa_node;
	}

	::iovec region() const
	{
		return _pool0;
	}
};

class heap_rc
{
	heap_rc_shared *_heap;

public:
	explicit heap_rc(heap_rc_shared *area)
		: _heap(area)
	{
	}

	~heap_rc()
	{
	}

	heap_rc(const heap_rc &) noexcept = default;

	heap_rc & operator=(const heap_rc &) = default;

	heap_rc_shared *operator->() const
	{
		return _heap;
	}
};

#endif
