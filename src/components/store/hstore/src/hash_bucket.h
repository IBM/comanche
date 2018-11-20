#ifndef _DAWN_HSTORE_HASH_BUCKET_H
#define _DAWN_HSTORE_HASH_BUCKET_H

#include "owner.h"
#include "content.h"

/*
 * hash_bucket, consisting of space for "owner" and "content".
 */

namespace impl
{
	template <typename Value>
		class hash_bucket
			: public owner
			, public content<Value>
		{
		public:
			explicit hash_bucket(owner owner_)
				: owner{owner_}
				, content<Value>()
			{}
			explicit hash_bucket()
				: hash_bucket(owner())
			{}
			/* A hash_bucket consists of two colocated pieces: owner and contents.
			* They do not move together, therefore copy or move of an entire
			* hash_bucket is an error.
			*/
			hash_bucket(const hash_bucket &) = delete;
			hash_bucket &operator=(const hash_bucket &) = delete;
			hash_bucket(hash_bucket &) = delete;
			hash_bucket &operator=(hash_bucket &) = delete;
		};
}

#endif
