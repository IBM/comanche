/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef COMANCHE_HSTORE_NUPM_TYPES_H
#define COMANCHE_HSTORE_NUPM_TYPES_H

#if USE_CC_HEAP == 1 || USE_CC_HEAP == 3
#else
#error USE_CC_HEAP type incompatible with nupm
#endif

#include "hstore_common.h"
#include "persister_nupm.h"

#include <cstring> /* strerror */

class region;

class hstore_nupm;

class region_closer
{
	std::shared_ptr<hstore_nupm> _mgr;
public:
	region_closer(std::shared_ptr<hstore_nupm> mgr_)
		: _mgr(mgr_)
	{}

	void operator()(region *) noexcept
	{
#if 0
		/* Note: There is not yet a way to close a region.  And when there is,
		 * the name may be close_region rather than region_close.
		 */
		_mgr->region_close(r);
#endif
	}
};

using open_pool_handle = std::unique_ptr<region, region_closer>;

using Persister = persister_nupm;

#endif
