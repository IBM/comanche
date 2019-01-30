#ifndef _COMANCHE_POOL_POBJ_H
#define _COMANCHE_POOL_POBJ_H

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
