#include "hop_hash.h"

/*
 * Hopscotch hash table - template Key, Value, and allocators
 */

/* Inteded to implement Hopscotch hashing
 * http://mcg.cs.tau.ac.il/papers/disc2008-hopscotch.pdf
 */

impl::no_near_empty_bucket::no_near_empty_bucket(
	bix_t bi_
	, std::size_t size_
	, const std::string &why_
)
	: std::range_error{
		"no_near_empty_bucket (index "
		+ std::to_string(bi_)
		+ " of "
		+ std::to_string(size_)
		+ " buckets) because "
		+ why_
	}
	, _bi(bi_)
{}

impl::move_stuck::move_stuck(bix_t bi_, std::size_t size_)
	: no_near_empty_bucket{bi_, size_, __func__}
{}

impl::table_full::table_full(bix_t bi_, std::size_t size_)
	: no_near_empty_bucket{bi_, size_, __func__}
{}
