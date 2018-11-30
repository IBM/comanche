/*
 * Hopscotch hash table - template Key, Value, and allocators
 */

#include <type_traits> /* is_base_of */

#include "perishable.h"
#include "persist_switch.h"
#include "segment_layout.h"

/*
 * ===== persist_map =====
 */

template <typename Allocator>
	impl::persist_map<Allocator>::persist_map(std::size_t n, const Allocator &av_)
		: _size_control()
		, _segment_count(
			n/base_segment_size == 0 ? 1U : segment_layout::log2(n/base_segment_size)
		)
		, _sc{}
	{
		using void_allocator_t = typename Allocator::template rebind<void>::other;
		_sc[0].bp =
			bucket_allocator_t(av_).address(
				*new (
						&*bucket_allocator_t(av_).allocate(
							base_segment_size
							, typename void_allocator_t::const_pointer()
							, "persist_control_segment_0"
						)
					)
					bucket_aligned_t[base_segment_size]
				);

		/* while are not enough allocated segments to hold n elements */
		for ( auto ix = 1U; ix != _segment_count._target; ++ix )
		{
			auto segment_size = base_segment_size<<(ix-1U);

			_sc[ix].bp =
				bucket_allocator_t(av_).address(
					*new (
						&*bucket_allocator_t(av_).allocate(
							segment_size
							, typename void_allocator_t::const_pointer()
							, "persist_control_segment_n"
						)
					)
					bucket_aligned_t[base_segment_size << (ix-1U)]
				);
		}

		_segment_count._actual = _segment_count._target;
		using ps_t =
			persist_switch<Allocator, std::is_base_of<persister, Allocator>::value>;
		ps_t::persist(av_, &_segment_count, (&_segment_count)+1U, "count");
		ps_t::persist(av_, &_size_control, (&_size_control)+1U, "size");
	}
