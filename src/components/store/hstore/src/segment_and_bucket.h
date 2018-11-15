#ifndef _DAWN_SEGMENT_AND_BUCKET_H
#define _DAWN_SEGMENT_AND_BUCKET_H

#include <cstddef> /* size_t */

namespace impl
{
	class segment_layout;

	class segment_and_bucket
	{
	public:
		using six_t = std::size_t; /* segment indexes (but uint8_t would do) */
		using bix_t = std::size_t; /* bucket indexes */
	private:
		six_t _si;
		bix_t _bi;
		static auto segment_size(six_t si) -> bix_t;
		static auto ix_low(bix_t ix) -> bix_t;
		static auto ix_high(bix_t ix) -> bix_t;
	public:
		explicit segment_and_bucket(six_t si_, bix_t bi_)
			: _si(si_)
			, _bi(bi_)
		{
		}
		explicit segment_and_bucket(bix_t ix);
		auto incr(const segment_layout &) -> segment_and_bucket &;
		auto incr_for_iterator() -> segment_and_bucket &;
		auto add_small(const segment_layout &, unsigned fwd) -> segment_and_bucket &;
		auto subtract_small(const segment_layout &, unsigned bkwd) -> segment_and_bucket &;
		six_t si() const { return _si; }
		bix_t bi() const { return _bi; }
		/* inverse of make_segment_and_bucket */
		bix_t index() const;
	};
}

bool operator==(const impl::segment_and_bucket &a, const impl::segment_and_bucket &b);
bool operator!=(const impl::segment_and_bucket &a, const impl::segment_and_bucket &b);

#endif
