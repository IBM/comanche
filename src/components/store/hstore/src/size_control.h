/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef _COMANCHE_HSTORE_SIZE_CONTROL_H
#define _COMANCHE_HSTORE_SIZE_CONTROL_H

#include "persist_atomic.h"
#include "test_flags.h" /* TEST_HSTORE_PERISHABLE */

#include <cassert>
#include <cstddef> /* size_t */

/* Tracking the size of the table
 */

namespace impl
{
	class size_control
	{
		/* unstable == 0 implies that size and all content "state" fields are valid and persisted
		 * if unstable != 0, restart must individually count the contents and refresh the content "state" fields
		 *
		 * The size_and_unstable field is 2^N times the actual size,
		 * and the lower N bits represents "unstable," not part of the size.
		 */
		persistent_atomic_t<std::size_t> _size_and_unstable;
		static constexpr unsigned N = 8;
		static constexpr std::size_t count_1 = 1U<<N;
		std::size_t destable_count() const { return _size_and_unstable & (count_1-1U); }
	public:
		size_control()
			: _size_and_unstable(0)
		{}
		void size_set_stable(std::size_t n) { _size_and_unstable = (n << N); }
		std::size_t size() const { assert( is_stable() ); return size_unstable(); }
		std::size_t size_unstable() const { return _size_and_unstable >> N; }
		std::size_t size_and_unstable() const { return _size_and_unstable; }
		void stabilize() { assert(destable_count() != 0); --_size_and_unstable; }
		void destabilize() { ++_size_and_unstable; assert(destable_count() != 0); }
		void decr() { _size_and_unstable -= count_1; }
		void incr() { _size_and_unstable += count_1; }
		bool is_stable() const { return destable_count() == 0; }
	};

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
