#ifndef _DAWN_SEGMENT_LAYOUT_H
#define _DAWN_SEGMENT_LAYOUT_H

#include "trace_flags.h"
#include <cstddef> /* size_t */
#include <limits> /* numeric_limits */

#if DEBUG_TRACE_LOCATE
#include <iostream>
#endif

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
		static constexpr unsigned log2_base_segment_size = 4U;
		static constexpr bix_t base_segment_size = (1U << log2_base_segment_size);
		virtual six_t segment_count() const = 0;
		static unsigned log2(unsigned long v)
		{
			return
				std::numeric_limits<unsigned long>::digits - unsigned(__builtin_clzl(v))
				;
		}
	};
}

#endif
