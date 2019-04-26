/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

/*
 * Hopscotch hash table debug
 */

#include <cassert>

#include "owner_debug.tcc"
#include "content_debug.tcc"

template <typename LockOwner, typename LockContent>
	impl::bucket_print<LockOwner, LockContent>::bucket_print(
		const std::size_t ct_
		, LockOwner &c_
		, LockContent &i_
	)
		: _ct(ct_)
		, _c{&c_}
		, _i{&i_}
	{
		assert(c_.index() == i_.index());
	}

/*
 * ===== hash_bucket =====
 */

template <typename LockOwner, typename LockContent>
	auto impl::operator<<(
		std::ostream &o_
		, const bucket_print<LockOwner, LockContent> &p_
	) -> std::ostream &
	{
		const auto &b = p_.sb().deref();
		return o_
			<< "( "
			<< dump<true>::make_owner_print(p_.bucket_count(), p_.lock_owner())
			<< " "
			<< b
			<< " )";
	}

template <typename TableBase>
	auto impl::operator<<(
		std::ostream &o_
		, const bucket_print
			<
				bypass_lock<typename TableBase::bucket_t, owner>
				, bypass_lock<typename TableBase::bucket_t, content<typename TableBase::bucket_t>>
			> &p_
	) -> std::ostream &
	{
		const auto &b = p_.sb().deref();
		auto lk_shared_owner = bypass_lock<typename TableBase::bucket_t, owner>(p_.index());
		return o_
			<< "( "
			<< dump<true>::make_owner_print(p_.bucket_count(), lk_shared_owner)
			<< " "
			<< b
			<< " )";
	}

#include "table_debug.tcc"
