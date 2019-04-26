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


#include "hop_hash.h"

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
