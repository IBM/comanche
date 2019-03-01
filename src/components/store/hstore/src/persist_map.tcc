/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#include <type_traits> /* is_base_of */

#include "perishable.h"
#include "segment_layout.h"

/*
 * ===== persist_map =====
 */

template <typename Allocator>
	impl::persist_map<Allocator>::persist_map(std::size_t n, Allocator av_)
		: _size_control()
		, _segment_count(
			/* The map tends to split when it is about 40% full.
			 * Triple the excpected object count when creating a segment count.
			 */
			((n*3U)/base_segment_size == 0 ? 1U : segment_layout::log2((3U * n)/base_segment_size))
		)
		, _sc{}
	{
		do_initial_allocation(av_);
	}

template <typename Allocator>
	void impl::persist_map<Allocator>::do_initial_allocation(Allocator av_)
	{
		using void_allocator_t = typename Allocator::template rebind<void>::other;

		if ( _segment_count._actual.is_stable() )
		{
			if ( _segment_count._actual.value() == 0 )
			{
				auto ptr =
					bucket_allocator_t(av_).allocate(
						base_segment_size
						, typename void_allocator_t::const_pointer()
						, "persist_control_segment_0"
					);
				new ( &*ptr ) bucket_aligned_t[base_segment_size];
				_sc[0].bp = ptr;
				_segment_count._actual.incr();
				av_.persist(&_segment_count, sizeof _segment_count);
			}

			/* while not enough allocated segments to hold n elements */
			for ( auto ix = _segment_count._actual.value(); ix != _segment_count._specified; ++ix )
			{
				auto segment_size = base_segment_size<<(ix-1U);

				auto ptr =
					bucket_allocator_t(av_).allocate(
						segment_size
						, typename void_allocator_t::const_pointer()
						, "persist_control_segment_n"
					);
				new (&*ptr) bucket_aligned_t[base_segment_size << (ix-1U)];
				_sc[ix].bp = ptr;
				_segment_count._actual.incr();
				av_.persist(&_segment_count, sizeof _segment_count);
			}

			av_.persist(&_size_control, sizeof _size_control);
		}
	}

template <typename Allocator>
	void impl::persist_map<Allocator>::reconstitute(Allocator av_)
	{
		auto av = bucket_allocator_t(av_);
		if ( ! _segment_count._actual.is_stable() || _segment_count._actual.value() != 0 )
		{
			auto ix = 0U;
			av.reconstitute(base_segment_size, _sc[ix].bp);
			++ix;

			/* restore segments beyond the first */
			for ( ; ix != _segment_count._actual.value_not_stable(); ++ix )
			{
				auto segment_size = base_segment_size<<(ix-1U);
				av.reconstitute(segment_size, _sc[ix].bp);
			}
			if ( ! _segment_count._actual.is_stable() )
			{
				/* restore the last, "junior" segment */
				auto segment_size = base_segment_size<<(ix-1U);
				av.reconstitute(segment_size, _sc[ix].bp);
			}

		}
	}
