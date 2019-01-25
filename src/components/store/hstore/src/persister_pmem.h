#ifndef _COMANCHE_HSTORE_PERSISTER_PMEM_H
#define _COMANCHE_HSTORE_PERSISTER_PMEM_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#include <libpmem.h>
#pragma GCC diagnostic pop

#include <cstddef>

class persister_pmem
{
public:
	void persist(const void *a, std::size_t sz) const
	{
		::pmem_persist(a, sz);
	}
};

#endif
