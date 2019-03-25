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


#ifndef COMANCHE_HSTORE_OPEN_POOL_H
#define COMANCHE_HSTORE_OPEN_POOL_H

#include "pool_path.h"

#include <utility> /* move */

/* Note: the distinction between tracked_pool and open_pool is needed only
 * because the IKVStore interface allows an "opened" pool to be deleted.
 */
class tracked_pool
	: protected pool_path
	{
	public:
		explicit tracked_pool(const pool_path &p_)
			: pool_path(p_)
		{}
		virtual ~tracked_pool() {}
		pool_path path() const { return *this; }
	};

template <typename Handle>
	class open_pool
		: public tracked_pool
	{
		Handle _pop;
	public:
		explicit open_pool(
			const pool_path &path_
			, Handle &&pop_
		)
			: tracked_pool(path_)
			, _pop(std::move(pop_))
		{}
		open_pool(const open_pool &) = delete;
		open_pool& operator=(const open_pool &) = delete;
#if 1
		/* session constructor and get_pool_regions only */
		auto *pool() const { return _pop.get(); }
#endif
	};

#endif
