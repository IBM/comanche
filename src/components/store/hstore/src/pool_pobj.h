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


#ifndef _COMANCHE_HSTORE_POOL_POBJ_H
#define _COMANCHE_HSTORE_POOL_POBJ_H

#pragma GCC diagnostic push
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <libpmemobj.h> /* PMEMobjpool */
#pragma GCC diagnostic pop

#include <cstdint> /* uint64_t */

class pool_pobj
{
	PMEMobjpool * _pool;

public:
	explicit pool_pobj(PMEMobjpool * pool_)
		: _pool(pool_)
	{}

	PMEMobjpool *pool() const
	{
		return _pool;
	}
};

#endif
