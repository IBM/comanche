/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef _COMANCHE_PERSIST_MAP_H
#define _COMANCHE_PERSIST_MAP_H

#include "bucket_aligned.h"
#include "hash_bucket.h"
#include "persist_fixed_string.h"
#include "persistent.h"
#include "persist_atomic.h"
#include "segment_layout.h"
#include "size_control.h"

#include <cstddef> /* size_t */

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
	public:
			using bucket_aligned_t = bucket_aligned<hash_bucket<value_type>>;
	private:
			using bucket_allocator_t =
				typename Allocator::template rebind<bucket_aligned_t>::other;
			using bucket_ptr = typename bucket_allocator_t::pointer;
	private:
			/* bucket indexes */
			using bix_t = segment_layout::bix_t;
			/* segment indexes */
			using six_t = segment_layout::six_t;

			struct segment_count
			{
				/* current segment count */
				persistent_atomic_t<six_t> _actual;
				/* desired segment count */
				persistent_atomic_t<six_t> _target;
				segment_count(six_t target_)
					: _actual(0)
					, _target(target_)
				{}
			};

			struct segment_control
			{
				persistent_t<bucket_ptr> bp;
				segment_control()
					: bp()
				{
				}
			};

			static constexpr six_t _segment_capacity = 32U;
			static constexpr unsigned log2_base_segment_size =
				segment_layout::log2_base_segment_size;
			static constexpr bix_t base_segment_size =
				segment_layout::base_segment_size;

			size_control _size_control;

			segment_count _segment_count;

			segment_control _sc[_segment_capacity];

		public:
			persist_map(std::size_t n, Allocator av);
			void do_initial_allocation(Allocator av);
			void reconstitute(Allocator av);
			friend class persist_controller<Allocator>;
		};
}

// template<> struct type_number<typename persist_map<Allocator>::bucket_aligned_t> { static constexpr uint64_t value = 5; };

#include "persist_map.tcc"

#endif
