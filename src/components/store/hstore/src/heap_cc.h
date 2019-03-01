/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef COMANCHE_HSTORE_ALLOCATOR_CC_H
#define COMANCHE_HSTORE_ALLOCATOR_CC_H

#include "bad_alloc_cc.h"
#include "persister_cc.h"

#include <algorithm> /* min */
#include <array>
#include <cstring> /* memset */
#include <cstddef> /* size_t, ptrdiff_t */
#include <sstream> /* ostringstream */
#include <string>

#if 0
#include <iostream>
#endif

class sbrk_alloc
{
	struct bound
	{
		char *_end;
		void set(char *e) noexcept { _end = e; }
		char *end() const noexcept { return _end; }
	};
	struct state /* persists */
	{
		void *_location; /* persists. contains its own expected address */
		unsigned _sw; /* persists. Initially 0. Toggles between 0 and 1 */
		char *_limit; /* persists */
		std::array<bound, 2U> _bounds; /* persists, depends on _sw */
		bound &current() { return _bounds[_sw]; }
		bound &other() { return _bounds[1U-_sw]; }
		char *begin() { return static_cast<char *>(static_cast<void *>(this+1)); }
		void swap() { _sw = 1U - _sw; }
		void *limit() const { return _limit; }
	};
	bound &current() { return _state->current(); }
	bound &other() { return _state->other(); }
	void swap() { _state->swap(); }
	state *_state;
	template <typename T>
		void persist(const T &) {}
	void restore() const
	{
		if ( _state->_location != &_state->_location )
		{
			std::ostringstream s;
			s << "cc_heap region mapped at " << &_state->_location << " but required to be at " << _state->_location;
			throw std::runtime_error{s.str()};
		}
		assert(_state->_sw < _state->_bounds.size());
	}
public:
	explicit sbrk_alloc(void *area, std::size_t sz)
		: _state(static_cast<state *>(area))
	{
		/* one-time initialization; assumes that initial bytes in area are zeros/nullptr */
		_state->_limit = static_cast<char *>(area) + sz;
		_state->_bounds[0].set(_state->begin());
		_state->_sw = 0;
		_state->_location = &_state->_location;
		persist(_state);
	}
	explicit sbrk_alloc(void *area)
		: _state(static_cast<state *>(area))
	{
		restore();
	}
	void *malloc(std::size_t sz)
	{
		/* round to double word */
		sz = (sz + 63UL) & ~63UL;
		if ( static_cast<std::size_t>(_state->_limit - current().end()) < sz )
		{
#if 0
			std::cerr << "Alloc " << sz << " failed, remaining " << _state->_limit - current().end() << "\n";
#endif
			return nullptr;
		}
		auto p = current().end();
		auto q = p + sz;
		other().set(q);
		persist(other());
		swap();
#if 0
		if ( (std::uintptr_t(1) << 20) < sz )
		{
			std::cerr << "Large alloc " << sz << " remaining " << _state->_limit - current().end() << "\n";
		}
		else if ( ( std::uintptr_t(p) >> 20U ) != ( std::uintptr_t(q) >> 20U ) )
		{
			/* small alloc, but crossed a 1MB line. */
			std::cerr << "Sample alloc " << sz << " remaining " << _state->_limit - current().end() << "\n";
		}
#endif
		persist(_state->_sw);
		return p;
	}
	void free(const void *) {}
	void *area() const { return _state; }
};

class heap_cc
	: public sbrk_alloc
{
public:
	explicit heap_cc(void *area, std::size_t sz)
		: sbrk_alloc(area, sz)
	{}
	explicit heap_cc(void *area)
		: sbrk_alloc(area)
	{}
};

#endif
