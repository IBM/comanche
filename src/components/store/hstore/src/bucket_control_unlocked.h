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


#ifndef _COMANCHE_HSTORE_BUCKET_CONTROL_UNLOCKED_
#define _COMANCHE_HSTORE_BUCKET_CONTROL_UNLOCKED_

#include "bucket_aligned.h"
#include <cstddef> /* size_t */

namespace impl
{
	template <typename Bucket>
		class bucket_control_unlocked
		{
		public:
			using bucket_type = Bucket;
			using bucket_aligned_t = bucket_aligned<Bucket>;
			using six_t = std::size_t;
			using bix_t = std::size_t;
		public:
			six_t _index;
		public:
			bucket_control_unlocked<Bucket> *_prev;
			bucket_control_unlocked<Bucket> *_next;
			bucket_aligned_t *_buckets;
			bucket_aligned_t *_buckets_end;
		public:
			bucket_control_unlocked(
				six_t index_
				, bucket_aligned_t *buckets_
			)
				: _index(index_)
				, _prev(this)
				, _next(this)
				, _buckets(buckets_)
				, _buckets_end(buckets_ + ( segment_layout::log2_base_segment_size << (index_ == 0U ? 0U : (index_-1U) ) ) )
			{}
			six_t index() const { return _index; }
			std::size_t segment_size() const { return _buckets_end - _buckets; }
			bucket_aligned_t &deref(bix_t bi) const { return _buckets[bi]; }

			void deconstitute()
			{
				for ( auto it = _buckets; it != _buckets_end; ++it )
				{
					typename bucket_type::content_type &c = *it;
					/* deconsititute key and value */
					if ( c.state_get() != bucket_type::FREE )
					{
						c.value().first.deconstitute();
						c.value().second.deconstitute();
					}
				}
			}
			template <typename Allocator>
				void reconstitute(Allocator av_)
				{
					for ( auto it = _buckets; it != _buckets_end; ++it )
					{
						typename bucket_type::content_type &c = *it;
						/* reconsititute key and value */
						if ( c.state_get() != bucket_type::FREE )
						{
							c.value().first.reconstitute(av_);
							c.value().second.reconstitute(av_);
						}
					}
				}
		};
}

#endif
