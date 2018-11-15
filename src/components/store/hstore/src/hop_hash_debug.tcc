/*
 * Hopscotch hash table debug
 */

#include "cond_print.h"

#include <cassert>
#include <cstddef> /* size_t */
#include <sstream> /* ostringstream */

/*
 * ===== owner =====
 */

template <typename TableBase, typename Lock>
	auto impl::operator<<(
		std::ostream &o_
		, const owner_print<TableBase, Lock> &t_
	) -> std::ostream &
	{
		const auto &w = t_.get_table().locate_owner(t_.sb());
		return o_
			<< "(owner "
			<< w.owned(t_.get_table().bucket_count(), t_.lock())
			<< ")";
	}

template <typename TableBase>
	auto impl::operator<<(
		std::ostream &o_
		, const owner_print<TableBase, bypass_lock<const owner>> &t_
	) -> std::ostream &
	{
		const auto &w = t_.get_table().locate_owner(t_.sb());
		bypass_lock<const owner> lk(w, t_.sb());
		return o_
			<< "(owner "
			<< w.owned(t_.get_table().bucket_count(), lk)
			<< ")";
	}

/*
 * ===== content =====
 */

template <typename Value>
	auto impl::content<Value>::is_clear() const noexcept -> bool
	{
		return _owner == owner_undefined;
	}

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

template <typename Value>
	void impl::content<Value>::owner_verify(content::owner_t owner_) const
	{
		if ( _owner != owner_ )
		{
			std::cerr << __func__ << " non-owner " << owner_ << " attempt to move " << owner_ << "\n";
		}
		assert(_owner == owner_);
	}

template <typename Value>
	void impl::content<Value>::owner_update(owner_t owner_delta)
	{
		_owner |= owner_delta;
	}

template <typename Value>
	auto impl::content<Value>::state_string() const -> std::string
	{
		return _state == FREE ? "FREE" : _state == IN_USE ? "IN_USE" : _state == ENTERING ? "ENTERING" : "EXITING";
	}

template <typename TableBase, typename LockOwner, typename LockContent>
	impl::bucket_print<TableBase, LockOwner, LockContent>::bucket_print(
		const TableBase &t_
		, LockOwner &c_
		, LockContent &i_
	)
		: _t(&t_)
		, _c{&c_}
		, _i{&i_}
	{
		assert(c_.index() == i_.index());
	}

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

/*
 * ===== bucket =====
 */

template <typename TableBase, typename LockOwner, typename LockContent>
	auto impl::operator<<(
		std::ostream &o_
		, const bucket_print<TableBase, LockOwner, LockContent> &p_
	) -> std::ostream &
	{
		const auto &t = p_.get_table();
		const auto &b = t.locate(p_.sb());
		return o_
			<< "( "
			<< make_owner_print(t, p_.lock_owner())
			<< " "
			<< b
			<< " )";
	}

template <typename TableBase>
	auto impl::operator<<(
		std::ostream &o_
		, const bucket_print
			<
				TableBase
				, bypass_lock<owner>
				, bypass_lock<content<typename TableBase::bucket_t>>
			> &p_
	) -> std::ostream &
	{
		const auto &t = p_.get_table();
		const auto &b = t.locate(p_.index());
		auto lk_shared_owner = bypass_lock<owner>(p_.index());
		return o_
			<< "( "
			<< make_owner_print(t, lk_shared_owner)
			<< " "
			<< b
			<< " )";
	}

#include "table_debug.tcc"
