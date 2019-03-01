/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
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
		};
}

#endif
