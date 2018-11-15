#ifndef _DAWN_PERSIST_MAP_H
#define _DAWN_PERSIST_MAP_H

#include "bucket.h"
#include "bucket_aligned.h"
#include "persist_fixed_string.h"
#include "persistent.h"
#include "persist_atomic.h"
#include "segment_layout.h"

#include <cstddef> /* size_t */
#include <limits> /* numeric_limits */

/* Persistent data for hstore.
 */

namespace impl
{
	template <typename Allocator>
		class persist_controller;

	template <typename Allocator>
		class persist_map
		{
			using value_type = typename Allocator::value_type;
			using bucket_aligned_t = bucket_aligned<bucket<value_type>>;
			using bucket_allocator_t =
				typename Allocator::template rebind<bucket_aligned_t>::other;
			using void_allocator_t =
				typename Allocator::template rebind<void>::other;
			using bucket_ptr = typename bucket_allocator_t::pointer;
			struct segment_control
			{
				persistent_t<bucket_ptr> bp;
				segment_control()
					: bp()
				{
				}
			};

			using bix_t = segment_layout::bix_t; /* sufficient for all bucket indexes */
			using six_t = segment_layout::six_t;
			static constexpr six_t _segment_capacity = 32U;
			static constexpr unsigned log2_base_segment_size = segment_layout::log2_base_segment_size;
			static constexpr bix_t base_segment_size = segment_layout::base_segment_size;

			struct size_control_t
			{
				/* unstable == 0 implies that size is valid and persisted
				 * if unstable != 0, restart must individually count the contents.
				 */
				persistent_atomic_t<unsigned>    unstable;
				persistent_atomic_t<std::size_t> size;
				size_control_t()
					: unstable(0)
					, size(0)
				{}
			} _size_control;

			struct segment_count_t
			{
				/* current segment count */
				persistent_atomic_t<six_t> _actual;
				/* desired segment count */
				persistent_atomic_t<six_t> _target;
				segment_count_t(six_t target_)
					: _actual(0)
					, _target(target_)
				{}
			} _segment_count;

			segment_control _sc[_segment_capacity];

		public:
			persist_map(std::size_t n, const Allocator &av);
			friend class persist_controller<Allocator>;
		};
}

#include "persist_map.tcc"

#endif
