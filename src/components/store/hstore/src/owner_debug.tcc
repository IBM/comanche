/*
 * Hopscotch hash owner debug
 */

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
		, const owner_print<TableBase, bypass_lock<typename TableBase::bucket_t, const owner>> &t_
	) -> std::ostream &
	{
		const auto &w = t_.get_table().locate_owner(t_.sb());
		bypass_lock<typename TableBase::bucket_t, const owner> lk(w, t_.sb());
		return o_
			<< "(owner owns"
			<< w.owned(t_.get_table().bucket_count(), lk)
			<< ")";
	}

