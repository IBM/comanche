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


#ifndef COMANCHE_HSTORE_HEAD_CO_H
#define COMANCHE_HSTORE_HEAD_CO_H

#include "persister_cc.h"

#include <libpmemobj.h> /* PMEMoid */

#include <array>
#include <cstddef> /* size_t, ptrdiff_t */
#include <sstream> /* ostringstream */
#include <string>

#if 0
#include <iostream>
#endif

class sbrk_offset_heap
{
	struct bound
	{
		std::size_t _end;
		void set(std::size_t e) noexcept { _end = e; }
		std::size_t end() const noexcept { return _end; }
	};

	PMEMoid _oid;
	unsigned _sw; /* persists. Initially 0. Toggles between 0 and 1 */
	std::size_t _start; /* start of heap area, relative to this */
	std::size_t _size;
	std::array<bound, 2U> _bounds; /* persists, depends on _sw */
	bound &current() { return _bounds[_sw]; }
	bound &other() { return _bounds[1U-_sw]; }
	void swap() { _sw = 1U - _sw; }
	std::size_t limit() const { return _size; }
	bool match(PMEMoid oid_) const { return _oid.pool_uuid_lo == oid_.pool_uuid_lo && _oid.off == oid_.off; }

	template <typename T>
		void persist(const T &) {}
	void restore(PMEMoid oid_) const
	{
		if ( ! match(oid_) )
		{
			std::ostringstream e;
			e << "Cannot restore heap oid " << std::hex << _oid.pool_uuid_lo << "." << _oid.off << " from pool with mismatched oid " << oid_.pool_uuid_lo << "." << oid_.off;
			throw std::runtime_error(e.str());
		}
		assert(_sw < _bounds.size());
	}
public:
	explicit sbrk_offset_heap(PMEMoid oid_, std::size_t sz_, std::size_t start_)
		: _oid{_oid}
		, _sw{_sw}
		, _start{_start}
		, _size{_size}
		, _bounds{_bounds}
	{
		if ( ! match(oid_) )
		{
			/* one-time initialization; assumes that initial bytes in area are zeros/nullptr */
			assert(start_ <= sz_); /* else not enough space to construct, let alone malloc */
#if 0
			std::cerr << "Create " << this << "." << _sw << " start_ " << start_ << " oid off " << oid_.off << "\n";
#endif
			_start = start_ + oid_.off;
#if 0
			std::cerr << "Create " << this << "." << _sw << " _start " << _start << " oid off " << oid_.off << "\n";
#endif
			_size = sz_;
			_bounds[0]._end = _start;
			_sw = 0;
			_oid = oid_;
			persist(*this);
#if 0
			std::cerr << "Create " << this << "." << _sw << " start " << _start << " size " << _size << " limit " << limit() << " end " << current()._end << "\n";
#endif
		}
		else
		{
			restore(oid_);
#if 0
			std::cerr << "Restore  in create" << this << "." << _sw << " start " << _start << " size " << _size << " limit " << limit() << " end " << current()._end << "\n";
#endif
		}
	}

	explicit sbrk_offset_heap(PMEMoid oid_)
		: _oid{_oid}
		, _sw{_sw}
		, _start{_start}
		, _size{_size}
		, _bounds{_bounds}
	{
		restore(oid_);
#if 0
		std::cerr << "Restore " << this << "." << _sw << " start " << _start << " size " << _size << " limit " << limit() << " end " << current()._end << "\n";
#endif
	}

	PMEMoid malloc(std::size_t sz)
	{
		std::cerr << "Malloc " << this << "." << _sw << " start " << _start << " sz " << sz << " limit " << limit() << " end " << current()._end << "\n";
		/* round to double word */
		sz = (sz + 7UL) & ~7UL;
		if ( static_cast<std::size_t>(limit() - current()._end) < sz ) { return PMEMoid{}; }
		auto p = current().end();
		auto q = p + sz;
		other().set(q);
		persist(other());
		swap();
		persist(_sw);
		assert(_start + p < 16382ULL << 20ULL);
		std::cerr << "Malloc " << this << "." << _sw << " start " << _start << " sz " << sz << " limit " << limit() << " end " << current()._end << "\n";
		return PMEMoid { _oid.pool_uuid_lo, _start + p };
	}

	void free(PMEMoid) {}
};

class heap_co
	: public sbrk_offset_heap
{
public:
	explicit heap_co(PMEMoid oid_, std::size_t sz_, std::size_t start_)
		: sbrk_offset_heap(oid_, sz_, start_)
	{}
	explicit heap_co(PMEMoid oid_)
		: sbrk_offset_heap(oid_)
	{}
};

#endif
