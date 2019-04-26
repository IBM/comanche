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


#ifndef _COMANCHE_HSTORE_PERSIST_MAP_H
#define _COMANCHE_HSTORE_PERSIST_MAP_H

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
	using segment_count_actual_t = value_unstable<segment_layout::six_t, 1>;

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

			/* bucket indexes */
			using bix_t = segment_layout::bix_t;
			/* segment indexes */
			using six_t = segment_layout::six_t;

			struct segment_count
			{
				/* current segment count */
				segment_count_actual_t _actual;
				/* desired segment count */
				persistent_atomic_t<six_t> _specified;
				segment_count(six_t specified_)
					: _actual(0)
					, _specified(specified_)
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
