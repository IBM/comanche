/*
 * Hopscotch hash table debug
 */

#include <cassert>

#include "owner_debug.tcc"
#include "content_debug.tcc"

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

/*
 * ===== hash_bucket =====
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
