#ifndef _COMANCHE_HSTORE_PERSISTER_NUPM_H
#define _COMANCHE_HSTORE_PERSISTER_NUPM_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#include <nupm/pm_lowlevel.h>
#pragma GCC diagnostic pop

#include <cstddef>

class persister_nupm
{
public:
	static void persist(const void *a, std::size_t sz)
	{
		nupm::mem_flush(a, sz);
	}
};

#endif
