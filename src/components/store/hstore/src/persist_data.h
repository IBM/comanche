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
			using allocator_type = AllocatorSegment;
			persist_data(std::size_t n, const AllocatorSegment &av)
				: persist_map<AllocatorSegment>(n, av)
				, persist_atomic<TypeAtomic>()
			{}
		};
}

#endif
