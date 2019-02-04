#ifndef COMANCHE_HSTORE_NUPM_TYPES_H
#define COMANCHE_HSTORE_NUPM_TYPES_H

#define USE_CC_HEAP 1

#include "hstore_common.h"
#include "persister_nupm.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#pragma GCC diagnostic ignored "-Weffc++"
#include <nupm/dax_map.h>
#pragma GCC diagnostic pop

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
#if 1
	region_closer()
		: _mgr()
	{}
#endif
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

#include "hstore_open_pool.h"

using Persister = persister_nupm;

#endif
