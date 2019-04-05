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

#include "rc_alloc_wrapper_lb.h"

#include <boost/icl/interval_set.hpp>
#include <memory>
#include <cstddef> /* size_t, ptrdiff_t */

class heap_rc_shared
{
	static constexpr std::size_t alignment = 64U;
	void *_addr;
	std::size_t _size;
	unsigned _numa_node;
	nupm::Rca_LB _heap;
	/* The set of reconstituted addresses. Only needed during recovery.
	 * Potentially large, so should be erased after recovery. But there
	 * is no mechanism to erase it yet.
	 */
	using alloc_set_t = boost::icl::interval_set<const char *>; /* std::byte_t in C++17 */
	alloc_set_t _reconstituted; /* std::byte_t in C++17 */
	static void *best_aligned(void *a, std::size_t sz_)
	{
		auto begin = reinterpret_cast<uintptr_t>(a);
		auto cursor = begin + sz_ - 1U;
		auto end = begin + sz_;

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

		return reinterpret_cast<void *>(cursor);
	}
public:
	heap_rc_shared(std::size_t sz_, unsigned numa_node_)
		: _addr(best_aligned(this + 1, sz_ - sizeof *this))
		, _size((static_cast<char *>(static_cast<void *>(this)) + sz_) - static_cast<char *>(_addr))
		, _numa_node(numa_node_)
		, _heap()
		, _reconstituted()
	{
		/* cursor now locates the best-aligned region in  */
		_heap.add_managed_region(_addr, _size, _numa_node);
	}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-self"
	heap_rc_shared()
		: _addr(this->_addr)
		, _size(this->_size)
		, _numa_node(this->_numa_node)
		, _heap()
		, _reconstituted()
	{
		_heap.add_managed_region(_addr, _size, _numa_node);
	}
#pragma GCC diagnostic pop

	heap_rc_shared(const heap_rc_shared &) = delete;
	heap_rc_shared &operator=(const heap_rc_shared &) = delete;

	void *alloc(std::size_t sz_)
	{
		/* allocation must be multiple of alignment */
		auto sz = (sz_ + alignment - 1U)/alignment * alignment;
		auto p = _heap.alloc(sz, _numa_node, alignment);
		return p;
	}

	void inject_allocation(const void * p_, std::size_t sz_)
	{
		auto sz = (sz_ + alignment - 1U)/alignment * alignment;
		/* NOTE: inject_allocation should take a const void* */
		_heap.inject_allocation(const_cast<void *>(p_), sz, _numa_node);
		{
			auto pc = static_cast<alloc_set_t::element_type>(p_);
			_reconstituted.add(alloc_set_t::segment_type(pc, pc + sz));
		}
	}

	void free(void *p_, std::size_t sz_)
	{
		return _heap.free(p_, _numa_node, sz_);
	}

	bool is_reconstituted(const void * p_) const
	{
		return contains(_reconstituted, static_cast<alloc_set_t::element_type>(p_));
	}

	/* debug */
	unsigned numa_node() const
	{
		return _numa_node;
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

	heap_rc(const heap_rc &) = default;

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
};

#endif
