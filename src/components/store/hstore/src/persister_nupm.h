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
        PLOG("flush %p %lu", a, sz);
        count++;
      }
    }
#endif

	}
};

#endif
