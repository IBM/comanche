/*
 * Hopscotch hash table - template Key, Value, and allocators
 */

#include <type_traits> /* remove_const */
#include <utility> /* move */

#include "perishable.h"

/*
 * ===== content =====
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
template <typename Value>
	impl::content<Value>::content()
		: _state(FREE)
		, _v()
#if TRACK_OWNER
		, _owner(owner_undefined)
#endif
	{}
#pragma GCC diagnostic pop

template <typename Value>
	void impl::content<Value>::set_owner(owner_t
#if TRACK_OWNER
	       	owner_
#endif
	)
	{
#if TRACK_OWNER
		_owner = owner_;
#endif
	}

template <typename Value>
	auto impl::content<Value>::erase() -> void
	{
		if ( _state != FREE )
		{
			_v._value.~value_t();
			_state = FREE;
		}
		set_owner(owner_undefined);
	}

template <typename Value>
	auto impl::content<Value>::content_share(
		const content &sr_
		, std::size_t bi_
	) -> content &
	{
		using k_t = typename value_t::first_type;
		using m_t = typename value_t::second_type;
		new
			(&const_cast<std::remove_const_t<k_t> &>(_v._value.first))
			k_t(sr_._v._value.first)
			;
		new (&_v._value.second) m_t(sr_._v._value.second);
		set_owner(bi_);
		return *this;
	}

template <typename Value>
	auto impl::content<Value>::content_share(
		content &from_
	) -> content &
	{
		using k_t = typename value_t::first_type;
		using m_t = typename value_t::second_type;
		assert(_state == FREE);
		new
			(&const_cast<std::remove_const_t<k_t> &>(_v._value.first))
			k_t(from_._v._value.first)
			;
		new (&_v._value.second) m_t(from_._v._value.second);
		set_owner(from_.get_owner());
		return *this;
	}

template <typename Value>
	template <typename ... Args>
		auto impl::content<Value>::content_construct(
			std::size_t bi_
			, Args && ... args_
		) -> content &
		{
			assert(_state == FREE);
			new (&_v._value) Value(std::forward<Args>(args_)...);
			set_owner(bi_);
			return *this;
		}
