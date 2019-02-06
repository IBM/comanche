/*
 * Hopscotch hash table - template Key, Value, and allocators
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
#if 0
		std::cerr << "persist_map expected count " << n << ", scaled to " << (n*3U)
			<< ", requesting " << (3U * n)/base_segment_size
			<< " buckets requiring " << _segment_count._target << " segments" << "\n";
#endif
		do_initial_allocation(av_);
	}

template <typename Allocator>
	void impl::persist_map<Allocator>::do_initial_allocation(Allocator av_)
	{
#if 0
		std::cerr << "persist_map expected count " << n << ", scaled to " << (n*3U)
			<< ", requesting " << (3U * n)/base_segment_size
			<< " buckets requiring " << _segment_count._target << " segments" << "\n";
#endif
		using void_allocator_t = typename Allocator::template rebind<void>::other;

#if 0
		std::cerr << "persist_map::do_initial_allocation ENTER actual "
			<< _segment_count._actual << " target " << _segment_count._target << "\n";
#endif
		if ( _segment_count._actual == 0 )
		{
			{
				auto ptr =
					bucket_allocator_t(av_).allocate(
						base_segment_size
						, typename void_allocator_t::const_pointer()
						, "persist_control_segment_0"
					);
				new ( &*ptr ) bucket_aligned_t[base_segment_size];
				_sc[0].bp = ptr;
				_segment_count._actual = 1U;
				av_.persist(&_segment_count, sizeof _segment_count);
			}

			/* while not enough allocated segments to hold n elements */
			for ( auto ix = _segment_count._actual; ix != _segment_count._target; ++ix )
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
				_segment_count._actual = ix + 1U;
				av_.persist(&_segment_count, sizeof _segment_count);
			}

			av_.persist(&_size_control, sizeof _size_control);
		}
#if 0
		std::cerr << "persist_map::do_initial_allocation EXIT actual "
			<< _segment_count._actual << " target " << _segment_count._target << "\n";
#endif
	}
