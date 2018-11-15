#ifndef _DAWN_HSTORE_BUCKET_H
#define _DAWN_HSTORE_BUCKET_H

#include "owner.h"
#include "content.h"

/*
 * bucket, consisting of space for "owner" and "content".
 */

namespace impl
{
	template <typename Value>
		class bucket
			: public owner
			, public content<Value>
		{
		public:
			explicit bucket(owner c);
			explicit bucket();
			/* A bucket consists of two colocated pieces: owner and contents.
			* THey do not move together, therefore copy or move of an entire
			* bucket is an error.
			*/
			bucket(const bucket &) = delete;
			bucket &operator=(const bucket &) = delete;
			bucket(bucket &) = delete;
			bucket &operator=(bucket &) = delete;
		};
}

#endif
