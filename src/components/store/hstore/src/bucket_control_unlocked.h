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
#include "hop_hash_log.h"
#include "trace_flags.h"
#include <array>
#include <cstddef> /* size_t */

namespace impl
{
	template <typename Bucket>
		class bucket_control_unlocked
		{
			unsigned bit_count_v(typename Bucket::owner_type::value_type v)
			{
				unsigned count = 0;
				for ( ; v; v &= (v - 1) )
				{
					++count;
				}
				return count;
			}
			void report()
			{
				/* Report the distribution of owned element counts for each owner.
				 * We do now wrap, so the first owner::size element counts will wrongly counted.
				 */
				std::array<unsigned, bucket_type::owner_type::size> h{};
				typename bucket_type::owner_type::value_type all_owners_mask = 0U;
				for ( auto it = _buckets; it != _buckets_end; ++it )
				{
					typename bucket_type::owner_type &o = *it;
					/* cheat: owner::value will take *any* reference as a Lock */
					int lock = 0;
					auto bit_count = bit_count_v(o.value(lock));
					++h[bit_count];
				}
				{
					std::ostringstream hs{};
					for ( const auto &hn : h )
					{
						hs << " " << hn;
					}
					hop_hash_log<TRACE_MANY>::write(index(), ":", hs.str());
				}
				/* Report the distribution of owned element counts in each ownership range.
				 * We do now wrap, so the first owner::size element counts will wrongly counted. */
				unsigned j[bucket_type::owner_type::size] = {};
				for ( auto it = _buckets; it != _buckets_end; ++it )
				{
					typename bucket_type::owner_type &o = *it;
					int lock = 0;
					/* cheat: owner::value will take *any* reference as a Lock */
					all_owners_mask >>= 1U;
					assert((all_owners_mask & o.value(lock)) == 0);
					all_owners_mask |= o.value(lock);
					auto bit_count = bit_count_v(all_owners_mask);
					++j[bit_count];
				}
				{
					std::ostringstream js{};
					for ( const auto &jn : j )
					{
						js << " " << jn;
					}
					hop_hash_log<TRACE_MANY>::write(index(), ":", js.str());
				}
			}

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
				, _buckets_end(
					_buckets
					? buckets_ + ( segment_layout::log2_base_segment_size << (index_ == 0U ? 0U : (index_-1U) ) )
					: _buckets
				)
			{}
			bucket_control_unlocked(const bucket_control_unlocked &) = delete;

			~bucket_control_unlocked()
			{
#if TRACE_MANY
				/* report statistics for in-use segments */
				if ( _buckets != _buckets_end )
				{
					report();
				}
#endif
				deconstitute();
			}

			bucket_control_unlocked operator=(const bucket_control_unlocked &) = delete;

			six_t index() const { return _index; }
			std::size_t segment_size() const { return _buckets_end - _buckets; }
			bucket_aligned_t &deref(bix_t bi) const { return _buckets[bi]; }
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
