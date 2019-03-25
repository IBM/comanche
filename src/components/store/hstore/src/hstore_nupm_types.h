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


#ifndef COMANCHE_HSTORE_NUPM_TYPES_H
#define COMANCHE_HSTORE_NUPM_TYPES_H

#if USE_CC_HEAP == 1 || USE_CC_HEAP == 3
#else
#error USE_CC_HEAP type incompatible with nupm
#endif

#include "hstore_common.h"
#include "persister_nupm.h"

#include <cstring> /* strerror */

class region;

class hstore_nupm;

class region_closer
{
	std::shared_ptr<hstore_nupm> _mgr;
public:
	region_closer(std::shared_ptr<hstore_nupm> mgr_)
		: _mgr(mgr_)
	{}

	void operator()(region *) noexcept
	{
#if 0
		/* Note: There is not yet a way to close a region.  And when there is,
		 * the name may be close_region rather than region_close.
		 */
		_mgr->region_close(r);
#endif
	}
};

using open_pool_handle = std::unique_ptr<region, region_closer>;

using Persister = persister_nupm;

#endif
