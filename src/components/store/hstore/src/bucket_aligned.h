#ifndef _DAWN_BUCKET_ALIGNED_H
#define _DAWN_BUCKET_ALIGNED_H

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
