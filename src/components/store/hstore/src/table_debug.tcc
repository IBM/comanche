/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

/*
 * Hopscotch hash table debug
 */

#include "cond_print.h"

#include <cstddef> /* size_t */
#include <set>
#include <vector>

/*
 * ===== table =====
 */

template <typename TableBase>
	auto impl::operator<<(
		std::ostream &o_
		, const table_print<TableBase> &t_
	) -> std::ostream &
	{
		auto &tbl = t_.get_table();
		for ( const auto &k : tbl )
		{
			o_ << cond_print(k.first, "(key)") << " -> "
				<< cond_print(k.second, "(mapped)") << "\n";
		}
		return o_;
	}

template <typename TableBase>
	auto impl::operator<<(
		std::ostream &o_
		, const table_dump<TableBase> &t_
	) -> std::ostream &
	{
		auto &tbl_base = t_.get_table();
		o_ << "Buckets\n";
		std::set<std::size_t> owners;
		std::set<std::size_t> contents;
		std::size_t in_use_count = 0;
		for ( std::size_t k = 0; k != tbl_base.bucket_count(); ++k )
		{
			auto sb = tbl_base.make_segment_and_bucket(k);
			bypass_lock<typename TableBase::bucket_t, const owner> owner_lk(tbl_base.locate_owner(sb), sb);
			bypass_lock<typename TableBase::bucket_t, const content<typename TableBase::value_type>>
				content_lk(
					static_cast<const typename TableBase::content_t &>(sb.deref())
					, sb
				);

			auto v = owner_lk.ref().value(owner_lk);
			if (
				v != 0
				||
				! content_lk.ref().is_clear()
			)
			{
				o_ << k;
				for ( auto m = k; v != 0; v >>= 1, m = (m + 1) % tbl_base.bucket_count() )
				{
					/* v claims ownership of m */
					if ( v & 1 )
					{
						if ( ! owners.insert(m).second )
						{
							o_ << "MULTIPLE OWNERS for " << m << " ";
						}
						else
						{
						}
					}
				}	
				o_ << ": "
					<< make_bucket_print(tbl_base.bucket_count(), owner_lk, content_lk)
					<< "\n";
				if ( content_lk.ref().state_get() == TableBase::bucket_t::IN_USE ) { contents.insert(k); }
			}
		}

		if ( contents.size() != owners.size() )
		{
			o_ << "MISMATCH: content count " << in_use_count << " owner count " << owners.size();
			{
				o_ << " extra content ";
				std::vector<std::size_t> ec;
				std::set_difference(contents.begin(), contents.end(), owners.begin(), owners.end(), std::back_inserter(ec));
				for ( auto e : ec )
				{
					o_ << e << " ";
				}
			}
			{
				o_ << " extra owners ";
				std::vector<std::size_t> ec;
				std::set_difference(owners.begin(), owners.end(), contents.begin(), contents.end(), std::back_inserter(ec));
				for ( auto e : ec )
				{
					o_ << e << " ";
				}
			}

			o_ << "\n";
		}

		if ( ! tbl_base.segment_count_actual().is_stable() )
		{
			auto &loc = tbl_base._bc[tbl_base.segment_count()];
			if ( loc._buckets )
			{
				o_ << "Pending buckets\n";
				for ( std::size_t ks = 0; ks != tbl_base.bucket_count(); ++ks )
				{
					const auto kj = tbl_base.bucket_count() + ks;
					const auto sbj = tbl_base.make_segment_and_bucket(kj);
					bypass_lock<typename TableBase::bucket_t, const owner> owner_lk(loc._buckets[ks], sbj);
					bypass_lock<typename TableBase::bucket_t, const content<typename TableBase::value_type>>
						content_lk(
							loc._buckets[ks]
							, sbj
						);
					if (
						owner_lk.ref().value(owner_lk) != 0
						||
						! content_lk.ref().is_clear()
					)
					{
						o_ << kj << ": "
							<< make_bucket_print(tbl_base.bucket_count(), owner_lk, content_lk)
							<< "\n";
					}
				}
			}
			else
			{
				o_ << "Resize in progress but no pending buckets\n";
			}
		}
		return o_;
	}
