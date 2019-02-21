/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef COMANCHE_HSTORE_TRACKED_POOL_H
#define COMANCHE_HSTORE_TRACKED_POOL_H

#include "pool_path.h"

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

#endif
