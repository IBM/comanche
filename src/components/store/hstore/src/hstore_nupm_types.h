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

#include "hstore_config.h"

#if USE_CC_HEAP == 1 || USE_CC_HEAP == 3
#else
#error USE_CC_HEAP type incompatible with nupm
#endif

#include "hstore_common.h"
#include "persister_nupm.h"

#include <cstring> /* strerror */
#include <memory>

template <typename Region, typename Table, typename Allocator, typename LockType>
	class hstore_nupm;

template <typename Store, typename Region>
	class region_closer
	{
		std::shared_ptr<Store> _mgr;
	public:
		region_closer(std::shared_ptr<Store> mgr_)
			: _mgr(mgr_)
		{}

		void operator()(Region *) noexcept
		{
#if 0
			/* Note: There is not yet a way to close a region. And when there is,
			 * the name may be close_region rather than region_close.
			 */
			_mgr->region_close(r);
#endif
		}
	};

using Persister = persister_nupm;
#endif
