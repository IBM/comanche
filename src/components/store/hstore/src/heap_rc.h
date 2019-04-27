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

#include "bad_alloc_cc.h"

#include "histogram_log2.h"
#include "hop_hash_log.h"
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

class heap_rc_shared
{
	static constexpr unsigned log_alignment = 6U;
	static constexpr std::size_t alignment = 64U;
	void *_pool;
	std::size_t _size;
	unsigned _ref_count;
	unsigned _numa_node;
	nupm::Rca_LB _heap;
	util::histogram_log2<std::size_t> _hist_alloc;
	util::histogram_log2<std::size_t> _hist_inject;
	util::histogram_log2<std::size_t> _hist_free;
	/* Rca_LB seems not to allocate at or above about 2GiB. Limit reporting to 16 GiB. */
	static constexpr unsigned hist_report_upper_bound = 34U;
	/* The set of reconstituted addresses. Only needed during recovery.
	 * Potentially large, so should be erased after recovery. But there
	 * is no mechanism to erase it yet.
	 */
	using alloc_set_t = boost::icl::interval_set<const char *>; /* std::byte_t in C++17 */
	alloc_set_t _reconstituted; /* std::byte_t in C++17 */
	std::size_t _allocated;
	std::size_t _used;
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
public:
	heap_rc_shared(std::size_t sz_, unsigned numa_node_)
		: _pool(best_aligned(this + 1, sz_ - sizeof *this))
		, _size((static_cast<char *>(static_cast<void *>(this)) + sz_) - static_cast<char *>(_pool))
		, _ref_count(1)
		, _numa_node(numa_node_)
		, _heap()
		, _hist_alloc()
		, _hist_inject()
		, _hist_free()
		, _reconstituted()
		, _allocated(0)
		, _used(0)
	{
		/* cursor now locates the best-aligned region */
		_heap.add_managed_region(_pool, _size, _numa_node);
		hop_hash_log<TRACE_HEAP>::write(__func__, " this ", this, " size ", _size, " new");
		VALGRIND_CREATE_MEMPOOL(_pool, 0, false);
	}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-self"
	heap_rc_shared()
		: _pool(this->_pool)
		, _size(this->_size)
		, _ref_count(1)
		, _numa_node(this->_numa_node)
		, _heap()
		, _hist_alloc()
		, _hist_inject()
		, _hist_free()
		, _reconstituted()
		, _allocated(0)
		, _used(0)
	{
		_heap.add_managed_region(_pool, _size, _numa_node);
		hop_hash_log<TRACE_HEAP>::write(__func__, " this ", this, " size ", _size, " reconstituting");
		VALGRIND_MAKE_MEM_DEFINED(_pool, _size);
		VALGRIND_CREATE_MEMPOOL(_pool, 0, true);
	}
#pragma GCC diagnostic pop

	~heap_rc_shared()
	{
		hop_hash_log<TRACE_HEAP>::write(__func__, " this ", this, " size ", _size, " allocated ", _allocated, " used ", _used);
		VALGRIND_DESTROY_MEMPOOL(_pool);
		VALGRIND_MAKE_MEM_UNDEFINED(_pool, _size);
		write_hist<TRACE_HEAP_SUMMARY>();
	}

	heap_rc_shared(const heap_rc_shared &) = delete;
	heap_rc_shared &operator=(const heap_rc_shared &) = delete;

	template <bool B>
		void write_hist() const
		{
			hop_hash_log<B>::write(__func__, " pool ", _pool);
			std::size_t lower_bound = 0;
			for ( unsigned i = std::max(0U, log_alignment); i != std::min(std::size_t(hist_report_upper_bound), _hist_alloc.data().size()); ++i )
			{
				const std::size_t upper_bound = 1ULL << i;
				hop_hash_log<B>::write(__func__, " [", lower_bound, "..", upper_bound, "): ", _hist_alloc.data()[i], " ", _hist_inject.data()[i], " ", _hist_free.data()[i], " ");
				lower_bound = upper_bound;
			}
		}

	void *alloc(std::size_t sz_)
	{
		/* allocation must be multiple of alignment */
		auto sz = (sz_ + alignment - 1U)/alignment * alignment;

		try {
			auto p = _heap.alloc(sz, _numa_node, alignment);
					/* Note: allocation exception from Rca_LB is General_exception, which does not derive
					 * from std::bad_alloc.
					 */

			VALGRIND_MEMPOOL_ALLOC(_pool, p, sz);
			hop_hash_log<TRACE_HEAP>::write(__func__, " pool ", _pool, " addr ", p, " size ", sz_, " -> ", sz);
			_used += sz_;
			_allocated += sz;
			_hist_alloc.enter(sz);
			return p;
		}
		catch ( const std::bad_alloc & )
		{
			write_hist<true>();
			throw;
		}
		catch ( const General_exception & )
		{
			write_hist<true>();
			throw;
		}
	}

	void inject_allocation(const void * p, std::size_t sz_)
	{
		auto sz = (sz_ + alignment - 1U)/alignment * alignment;
		/* NOTE: inject_allocation should take a const void* */
		_heap.inject_allocation(const_cast<void *>(p), sz, _numa_node);
		VALGRIND_MEMPOOL_ALLOC(_pool, p, sz);
		hop_hash_log<TRACE_HEAP>::write(__func__, " pool ", _pool, " addr ", p, " size ", sz);

		{
			auto pc = static_cast<alloc_set_t::element_type>(p);
			_reconstituted.add(alloc_set_t::segment_type(pc, pc + sz));
		}
		_used += sz_;
		_allocated += sz;
		_hist_inject.enter(sz);
	}

	void free(void *p_, std::size_t sz_)
	{
		auto sz = (sz_ + alignment - 1U)/alignment * alignment;
		VALGRIND_MEMPOOL_FREE(_pool, p_);
		hop_hash_log<TRACE_HEAP>::write(__func__, " pool ", _pool, " addr ", p_, " size ", sz);
		_used -= sz_;
		_allocated -= sz;
		_hist_free.enter(sz);
		return _heap.free(p_, _numa_node, sz);
	}

	bool is_reconstituted(const void * p_) const
	{
		return contains(_reconstituted, static_cast<alloc_set_t::element_type>(p_));
	}

	heap_rc_shared &inc_ref()
	{
		++_ref_count;
		assert(_ref_count != 0 );
		return *this;
	}

	unsigned dec_ref()
	{
		assert(_ref_count != 0 );
		return --_ref_count;
	}

	/* debug */
	unsigned numa_node() const
	{
		return _numa_node;
	}

    ::iovec region() const
    {
      return ::iovec{_pool, _size};
    }
};

class heap_rc
{
	heap_rc_shared *_heap;
public:
	explicit heap_rc(void *area, std::size_t sz_, unsigned numa_node_)
		: _heap(new (area) heap_rc_shared(sz_, numa_node_))
	{
	}

	explicit heap_rc(void *area)
		: _heap(new (area) heap_rc_shared())
	{
	}

	~heap_rc()
	{
		auto r = _heap->dec_ref();
		if ( r == 0 )
		{
			_heap->~heap_rc_shared();
		}
	}

	heap_rc(const heap_rc &o_) noexcept
		: _heap(&o_._heap->inc_ref())
	{
	}

	heap_rc & operator=(const heap_rc &) = default;

	void *alloc(std::size_t sz_)
	{
		return _heap->alloc(sz_);
	}

	void inject_allocation(const void * p_, std::size_t sz_)
	{
		return _heap->inject_allocation(p_, sz_);
	}

	void free(void *p_, std::size_t sz_)
	{
		return _heap->free(p_, sz_);
	}

	bool is_reconstituted(const void *p_)
	{
		return _heap->is_reconstituted(p_);
	}

    ::iovec region() const
    {
      return _heap->region();
    }
};

#endif
