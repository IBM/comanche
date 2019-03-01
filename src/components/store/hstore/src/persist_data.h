/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef _COMANCHE_HSTORE_PERSIST_DATA_H
#define _COMANCHE_HSTORE_PERSIST_DATA_H

#include "persist_atomic.h"
#include "persist_map.h"

/* Persistent data for hstore.
 *  - persist_map: anchors for the unordered map
 *  - persist_atomic: currently in-progress atomic operation, if any
 */

namespace impl
{
	template <typename AllocatorSegment, typename TypeAtomic>
		class persist_data
			: public persist_map<AllocatorSegment>
			, public persist_atomic<TypeAtomic>
		{
		public:
			persist_data(std::size_t n, const AllocatorSegment &av)
				: persist_map<AllocatorSegment>(n, av)
				, persist_atomic<TypeAtomic>()
			{}
		};
}

#endif
