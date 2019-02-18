/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef COMANCHE_HSTORE_OPEN_POOL_H
#define COMANCHE_HSTORE_OPEN_POOL_H

#include "pool_path.h"

#include <utility> /* move */

/* Note: the distinction between tracked_pool and open_pool is needed only
 * because the IKVStore interface allows an "opened" pool to be deleted.
 */
class tracked_pool
	: protected pool_path
	{
	public:
		explicit tracked_pool(const pool_path &p_)
			: pool_path(p_)
		{}
		virtual ~tracked_pool() {}
		pool_path path() const { return *this; }
	};

template <typename Handle>
	class open_pool
		: public tracked_pool
	{
		Handle _pop;
	public:
		explicit open_pool(
			const pool_path &path_
			, Handle &&pop_
		)
			: tracked_pool(path_)
			, _pop(std::move(pop_))
		{}
		open_pool(const open_pool &) = delete;
		open_pool& operator=(const open_pool &) = delete;
#if 1
		/* session constructor and get_pool_regions only */
		auto *pool() const { return _pop.get(); }
#endif
	};

#endif
