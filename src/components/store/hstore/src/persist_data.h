#ifndef _DAWN_PERSIST_DATA_H
#define _DAWN_PERSIST_DATA_H

#include "persist_atomic.h"
#include "persist_map.h"

/* Persistent data for hstore.
 */

namespace impl
{
	template <typename AllocatorSegment, typename AllocatorAtomic>
		class persist_data
			: public persist_map<AllocatorSegment>
			, public persist_atomic<AllocatorAtomic>
		{
		public:
			persist_data(std::size_t n, const AllocatorSegment &av)
				: persist_map<AllocatorSegment>(n, av)
				, persist_atomic<AllocatorAtomic>()
			{}
		};
}

#endif
