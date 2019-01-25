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
	std::uint64_t _type_num;
public:

	explicit pool_pobj(PMEMobjpool * pool_, std::uint64_t type_num_)
		: _pool(pool_)
		, _type_num(type_num_)
	{}

	PMEMobjpool *pool() const
	{
		return _pool;
	}

	/* Note the type num is a function of the type,
	 * and should be tracked by the allocator, not here.
	 */
	std::uint64_t type_num() const
	{
		return _type_num;
	}
};

#endif
