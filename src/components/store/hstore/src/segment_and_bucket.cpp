/*
 * Hopscotch hash table - template Key, Value, and allocators
 */

#include "segment_and_bucket.h"

#include "segment_layout.h"

/*
 * ===== segment_and_bucket =====
 */

auto impl::segment_and_bucket::ix_low(bix_t ix) -> bix_t
{
	return ix % segment_layout::base_segment_size;
}

auto impl::segment_and_bucket::ix_high(bix_t ix) -> bix_t
{
	return ix >> segment_layout::log2_base_segment_size;
}

impl::segment_and_bucket::segment_and_bucket(bix_t ix)
	: _si(
		__builtin_expect((ix_high(ix) == 0),false)
		? 0
		: segment_layout::log2(ix_high(ix))
	)
	, _bi(
		(
			__builtin_expect((ix_high(ix) == 0),false)
			? 0
			: ix_high(ix) % (1U << (_si-1))
		)
		*
		segment_layout::base_segment_size + ix_low(ix)
	)
{
}

auto impl::segment_and_bucket::index() const -> bix_t
{
	/* recover original index */
	return
		(
			_si == 0U
			? 0U
			: ( segment_layout::base_segment_size << ( _si - 1U ) )
		)
		+ _bi
	;
}

/* number of buckets in a segment */
auto impl::segment_and_bucket::segment_size(six_t si) -> bix_t
{
	return segment_layout::base_segment_size << ( ( si == 0 ? 1U : si ) - 1U );
}

auto impl::segment_and_bucket::incr(const segment_layout &sl_) -> segment_and_bucket &
{
	/* To develop (six_t, bix_t) pair:
	 *  1. Increment the bix_t (low) part.
	 *  2. In case there was carry (next address is in a following segment)
	 *    a. Increment the six_t (high) part
	 *    b. In case of carry, wrap
	 */
	_bi = (_bi + 1U) % segment_size(_si);
	if ( _bi == 0U )
	{
		_si = (_si + 1U) % sl_.segment_count();
	}
	return *this;
}

auto impl::segment_and_bucket::add_small(
	const segment_layout &sl_
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
	_bi = (_bi + fwd) % segment_size(_si);
	if ( _bi < old_bi )
	{
		_si = (_si + 1U) % sl_.segment_count();
	}
	return *this;
}

auto impl::segment_and_bucket::subtract_small(
	const segment_layout &sl_
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
		_si = ( _si == 0 ? sl_.segment_count() : _si ) - 1U;
		_bi %= segment_size(_si);
	}
	else
	{
		_bi -= bkwd;
	}
	return *this;
}

auto impl::segment_and_bucket::incr_for_iterator() -> segment_and_bucket &
{
	/* To develop (six_t, bix_t) pair:
	 *  1. Increment the bix_t (low) part.
	 *  2. In case there was carry (next address is in following segment)
	 *    a. Increment the six_t (high) part
	 *    b. In case of carry, wrap
	 */
	_bi = (_bi + 1U) % segment_size(_si);
	if ( _bi == 0U )
	{
		++_si;
	}
	return *this;
}

bool operator==(
	const impl::segment_and_bucket &a_
	, const impl::segment_and_bucket &b_
)
{
	/* (test bi first as it is the more likely mismatch) */
	return a_.bi() == b_.bi() && a_.si() == b_.si();
}

bool operator!=(
	const impl::segment_and_bucket &a_
	, const impl::segment_and_bucket &b_
)
{
	return ! ( a_ == b_);
}
