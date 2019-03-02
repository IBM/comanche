/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef _COMANCHE_HSTORE_VALUE_UNSTABLE_H
#define _COMANCHE_HSTORE_VALUE_UNSTABLE_H

#include "persist_atomic.h"

#include <cassert>

/* A counter which may at times be "unstable" - part way between two values.
 */

namespace impl
{
	template <typename T, unsigned N>
		class value_unstable
		{
			/* unstable == 0 implies that value and all content "state" fields are valid and persisted
			 * if unstable != 0, restart must individually count the contents and refresh the content "state" fields
			 *
			 * The value_and_unstable field is 2^N times the actual value,
			 * and the lower N bits represents "unstable," not part of the value.
			 */
			persistent_atomic_t<T> _value_and_unstable;
			static constexpr T count_1 = 1U<<N;
			T destable_count() const { return _value_and_unstable & (count_1-1U); }
		public:
			value_unstable(T v)
				: _value_and_unstable(v)
			{}
			value_unstable()
				: value_unstable(0)
			{}
			void value_set_stable(T n) { _value_and_unstable = (n << N); }
			T value() const { assert( is_stable() ); return value_not_stable(); }
			T value_not_stable() const { return _value_and_unstable >> N; }

			void stabilize() { assert(destable_count() != 0); --_value_and_unstable; }
			void destabilize() { ++_value_and_unstable; assert(destable_count() != 0); }
			void decr() { _value_and_unstable -= count_1; }
			void incr() { _value_and_unstable += count_1; }
			bool is_stable() const { return destable_count() == 0; }
		};

	template <typename T, unsigned N>
		constexpr T value_unstable<T, N>::count_1;
}

#endif
