/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef _COMANCHE_HSTORE_BUCKET_ALIGNED_H
#define _COMANCHE_HSTORE_BUCKET_ALIGNED_H

/*
 * A 64-byte aligned holder for data.
 */

namespace impl
{
	template <typename Data>
		struct alignas(64) bucket_aligned
			: public Data
		{
			using value_type = Data;
			bucket_aligned()
				: Data()
			{}
		};
}

#endif
