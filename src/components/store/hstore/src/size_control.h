/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


#ifndef _COMANCHE_HSTORE_SIZE_CONTROL_H
#define _COMANCHE_HSTORE_SIZE_CONTROL_H

#include "persist_atomic.h"
#include "value_unstable.h"
#include "test_flags.h" /* TEST_HSTORE_PERISHABLE */

#include <cassert>
#include <cstddef> /* size_t */

/* Tracking the size of the table
 */

namespace impl
{
	using size_control = value_unstable<std::size_t, 8>;

	/* Size change state as an object. Not for RIAA strictness, but to move a memory
	 * fetch which appears to stall the pipeline following a call to pmem_persist.
	 */
	class size_change
	{
		size_control *_size_control;
		size_change(const size_change &) = delete;
		size_change& operator=(const size_change &) = delete;
	protected:
		size_control & get_size_control() const { return *_size_control; }
		virtual ~size_change() {}
	public:
		size_change(size_control &ctl)
			: _size_control(&ctl)
		{
		}
		virtual void change() const = 0;
	};

	class size_incr
		: public size_change
	{
	public:
		size_incr(size_control &ctl)
			: size_change(ctl)
		{}
		void change() const { get_size_control().incr(); }
	};

	class size_decr
		: public size_change
	{
	public:
		size_decr(size_control &ctl)
			: size_change(ctl)
		{}
		void change() const { get_size_control().decr(); }
	};

	class size_no_change
		: public size_change
	{
	public:
		size_no_change(size_control &ctl)
			: size_change(ctl)
		{}
		void change() const { }
	};
}

#endif
