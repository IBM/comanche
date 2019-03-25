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


#ifndef _COMANCHE_HSTORE_SEGMENT_LAYOUT_H
#define _COMANCHE_HSTORE_SEGMENT_LAYOUT_H

#include <cstddef> /* size_t */
#include <limits> /* numeric_limits */

/* Size and number of segments
 */

namespace impl
{
	class segment_layout
	{
	public:
		/* all bucket indexes */
		using bix_t = std::size_t;
		/* segment indexes (but uint8_t woud do) */
		using six_t = std::size_t;
	protected:
		~segment_layout() {}
	public:
		static constexpr unsigned log2_base_segment_size = 7U;
		static constexpr bix_t base_segment_size = (1U << log2_base_segment_size);
		virtual six_t segment_count() const = 0;
		static unsigned log2(unsigned long v)
		{
			return
				std::numeric_limits<unsigned long>::digits - unsigned(__builtin_clzl(v))
				;
		}
		static auto ix_high(bix_t ix) -> bix_t
		{
			return ix >> log2_base_segment_size;
		}
		static auto ix_low(bix_t ix) -> bix_t
		{
			return ix % base_segment_size;
		}
	};
}

#endif
