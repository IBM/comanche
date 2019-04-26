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


#ifndef _COMANCHE_HSTORE_PERSISTER_NUPM_H
#define _COMANCHE_HSTORE_PERSISTER_NUPM_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#include <nupm/pm_lowlevel.h>
#include <libpmem.h>
#include <common/logging.h>
#pragma GCC diagnostic pop

#include <cstddef>

class persister_nupm
{
public:
	static void persist(const void *a, std::size_t sz)
	{
    nupm::mem_flush_nodrain(a,sz);
    //pmem_flush(a,sz);

#if 0
    {
      static int count =0;
      if(count < 500) {
        PLOG("flush %p %zu", a, sz);
        count++;
      }
    }
#endif

	}
};

#endif
