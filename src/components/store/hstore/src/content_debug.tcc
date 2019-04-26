/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

/*
 * Hopscotch hash content debug
 */

#include "cond_print.h"

#include <cassert>
#include <cstddef> /* size_t */
#include <sstream> /* ostringstream */

/*
 * ===== content =====
 */

/* NOTE: Not reliable if recovering from a crash */
template <typename Value>
	auto impl::content<Value>::is_clear() const noexcept -> bool
	{
		return _owner == owner_undefined;
	}

#if TRACED_CONTENT
template <typename Value>
	auto impl::content<Value>::to_string() const -> std::string
	{
		return
			is_clear()
			? "empty"
			: descr()
			;
	}

template <typename Value>
	auto impl::content<Value>::descr() const -> std::string
	{
		std::ostringstream s;
		s << "(owner " << _owner << " "
			<< cond_print(key(),"(unprintable key)")
			<< "->"
			<< cond_print(mapped(), "(unprintable mapped)")
			<< ")"
			;
		return s.str();
	}
#endif

template <typename Value>
	void impl::content<Value>::owner_verify(content::owner_t owner_) const
	{
		if ( _owner != owner_ )
		{
			hop_hash_log<TRACE_MANY>::write(__func__, " source owned by ", _owner, " was about to be moved by ", owner_);
		}
		assert(_owner == owner_);
	}

template <typename Value>
	void impl::content<Value>::owner_update(owner_t owner_delta)
	{
		_owner |= owner_delta;
	}

#if TRACED_CONTENT
template <typename Value>
	auto impl::content<Value>::state_string() const -> std::string
	{
		return
			_state == FREE ? "FREE"
			: _state == IN_USE ? "IN_USE"
			: "?"
			;
	}
#endif

template <typename Value>
	auto impl::operator<<(
		std::ostream &o_
		, const content<Value> &c_
	) -> std::ostream &
	{
		return o_
			<< c_.state_string()
			<< " "
			<< c_.to_string()
			;
	}
