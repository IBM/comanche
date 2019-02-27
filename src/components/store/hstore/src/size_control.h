/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
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
