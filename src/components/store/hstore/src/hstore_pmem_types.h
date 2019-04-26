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


#ifndef COMANCHE_HSTORE_PMEM_TYPES_H
#define COMANCHE_HSTORE_PMEM_TYPES_H

template <unsigned Offse>
	struct check_offset;
#if USE_CC_HEAP == 1
#elif USE_CC_HEAP == 2
template <>
    struct check_offset<0U>
    {
    };
#else /* USE_CC_HEAP */
template <>
    struct check_offset<48U>
    {
    };
#endif /* USE_CC_HEAP */
#include "allocator_pobj_cache_aligned.h"
#include "hstore_common.h"
#include "persister_pmem.h"

#pragma GCC diagnostic push
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <libpmemobj.h>
#pragma GCC diagnostic pop

using open_pool_handle = std::unique_ptr<PMEMobjpool, void(*)(PMEMobjpool *)>;

#include "hstore_open_pool.h"

using Persister = persister_pmem;

#endif
