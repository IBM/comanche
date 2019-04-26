/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

/*
 * Hopscotch hash owner debug
 */

/*
 * ===== owner =====
 */

template <typename Lock>
	auto impl::operator<<(
		std::ostream &o_
		, const owner_print<Lock> &op_
	) -> std::ostream &
	{
		const auto &w = static_cast<const owner &>(op_.sb().deref());
		return o_
			<< "(owner "
			<< w.owned(op_.bucket_count(), op_.lock())
			<< ")";
	}

template <typename TableBase>
	auto impl::operator<<(
		std::ostream &o_
		, const owner_print<bypass_lock<typename TableBase::bucket_t, const owner>> &op_
	) -> std::ostream &
	{
		const auto &w = static_cast<const owner &>(op_.sb().deref());
		bypass_lock<typename TableBase::bucket_t, const owner> lk(w, op_.sb());
		return o_
			<< "(owner owns"
			<< w.owned(op_.get_table().bucket_count(), lk)
			<< ")";
	}

