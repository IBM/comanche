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


#ifndef _COMANCHE_HSTORE_SEGMENT_AND_BUCKET_H
#define _COMANCHE_HSTORE_SEGMENT_AND_BUCKET_H

#include "bucket_control_unlocked.h"
#include "segment_layout.h"
#include <cstddef> /* size_t */
#include <ostream>

namespace impl
{
	template <typename Bucket>
		class segment_and_bucket
		{
		public:
			using six_t = std::size_t; /* segment indexes (but uint8_t would do) */
			using bix_t = std::size_t; /* bucket indexes */
		private:
			const bucket_control_unlocked<Bucket> *_seg;
			bix_t _bi;
			bix_t segment_size() const { return _seg->segment_size(); }
		public:
			explicit segment_and_bucket(const bucket_control_unlocked<Bucket> *seg_, bix_t bi_)
				: _seg(seg_)
				, _bi(bi_)
			{
			}
			auto &deref() const { return _seg->deref(_bi); }
			auto incr_with_wrap() -> segment_and_bucket &
			{
				/* To develop (six_t, bix_t) pair:
				 *  1. Increment the bix_t (low) part.
				 *  2. In case there was carry (next address is in a following segment)
				 *    a. Increment the six_t (high) part
				 *    b. In case of carry, wrap
				 */
				_bi = (_bi + 1U) % segment_size();
				if ( _bi == 0U )
				{
					_seg = _seg->_next;
				}
				return *this;
			}
			/* one or the other is necessary, not both */
			bool at_end() const { return _bi == segment_size() && _seg->_next->index() == 0; }
			bool can_incr_without_wrap() const { return _bi != segment_size() || _seg->_next->index() != 0; }

			auto incr_without_wrap() -> segment_and_bucket &
			{
				/* To develop (six_t, bix_t) pair:
				 *  1. Increment the bix_t (low) part.
				 *  2. In case there was carry (next address is in following segment)
				 *    a. Increment the six_t (high) part *unless doing so would wrap
				 *    b. In case of carry, do not wrap
				 */
				++_bi;
				if ( _bi == segment_size() && _seg->_next->index() != 0 )
				{
					_bi = 0;
					_seg = _seg->_next;
				}
				return *this;
			}
			auto add_small(
				const segment_layout &
				, unsigned fwd
			) -> segment_and_bucket &
				{
				/* To develop (six_t, bix_t) pair:
				 *  1. Add to the (low) part.
				 *  2. In case there was carry (next address is in following segment)
				 *    a. Increment the six_t (high) part
				 *    b. In case of carry, wrap
				 */
				const auto old_bi = _bi;
				_bi = (_bi + fwd) % segment_size();
				if ( _bi < old_bi )
				{
					_seg = _seg->_next;
				}
				return *this;
			}
			auto subtract_small(
				const segment_layout &
				, unsigned bkwd
			) -> segment_and_bucket &
			{
				/* To develop (six_t, bix_t) pair:
				 *  1. decrement the part.
				 *  2. In case there was borrow (next address is in previous segment)
				 *    a. decrement the six_t (high) part
				 *    b. wrap
				 *    c. remove high bits, a result of unsigned borrow
				 */
				if ( _bi < bkwd )
				{
					_bi -= bkwd;
					_seg = _seg->_prev;
					_bi %= segment_size();
				}
				else
				{
					_bi -= bkwd;
				}
				return *this;
			}
			six_t si() const { return _seg->_index; }
			bix_t bi() const { return _bi; }
			/* inverse of make_segment_and_bucket */
			bix_t index() const
			{
				return
					(
						si() == 0U
						? 0U
						: ( segment_layout::base_segment_size << ( si() - 1U ) )
					)
					+ _bi
				;
			}
		};

	template <typename Bucket>
		auto operator<<(
			std::ostream &o_
			, const segment_and_bucket<Bucket> &b_
		) -> std::ostream &
		{
			return o_ << b_.si() << "." << b_.bi();
		}

	template <typename Bucket>
		auto operator==(
			const segment_and_bucket<Bucket> &a_
			, const segment_and_bucket<Bucket> &b_
		) -> bool
		{
			/* (test bi first as it is the more likely mismatch) */
			return a_.bi() == b_.bi() && a_.si() == b_.si();
		}

	template <typename Bucket>
		auto operator!=(
			const segment_and_bucket<Bucket> &a_
			, const segment_and_bucket<Bucket> &b_
		) -> bool
		{
			return ! ( a_ == b_);
		}
}

#endif
