#ifndef _DAWN_HSTORE_HOP_HASH_DEBUG_H
#define _DAWN_HSTORE_HOP_HASH_DEBUG_H

#include <cstddef> /* size_t */
#include <iostream>

namespace impl
{
	template <typename Value>
		class content;
}

namespace impl
{
	class owner;

	template <typename Referent>
		class bucket_ref;

	template <typename Referent>
		class bypass_lock
			: public bucket_ref<Referent>
		{
		public:
			bypass_lock(Referent &b_, std::size_t i_)
				: bucket_ref<Referent>(&b_, i_)
			{}
		};

	template <typename Value>
		auto operator<<(
			std::ostream &o
			, const impl::content<Value> &c
		) -> std::ostream &;

	template <
		typename TableBase
		, typename Lock
	>
		class owner_print
		{
			const TableBase *_t;
			Lock *_i;
		public:
			owner_print(const TableBase &t_, Lock &i_)
				: _t(&t_)
				, _i{&i_}
			{}
			const TableBase &get_table() const { return *_t; }
			Lock &lock() const { return *_i; }
			std::size_t index() const { return lock().index(); }
		};

	template <
		typename TableBase
		, typename Lock
	>
		auto make_owner_print(
			const TableBase &t_
			, Lock &lk_
		) -> owner_print<TableBase, Lock>
		{
			return owner_print<TableBase, Lock>(t_, lk_);
		}

	template <
		typename TableBase
		, typename Lock
	>
		auto operator<<(
			std::ostream &o
			, const owner_print<TableBase, Lock> &
		) -> std::ostream &;

	template <typename TableBase>
		auto operator<<(
			std::ostream &o
			, const owner_print<TableBase, bypass_lock<const owner>> &
		) -> std::ostream &;

	template <
		typename TableBase
		, typename LockOwner
		, typename LockContent
	>
		class bucket_print
		{
			const TableBase *_t;
			LockOwner *_c;
			LockContent *_i;
		public:
			bucket_print(const TableBase &t, LockOwner &w, LockContent &i);
			const TableBase &get_table() const { return *_t; }
			LockOwner &lock_owner() const { return *_c; }
			LockContent &lock() const { return *_i; }
			std::size_t index() const { return lock().index(); }
		};

	template <
		typename TableBase
	>
		class table_print
		{
			const TableBase *_t;
		public:
			table_print(const TableBase &t_)
				: _t(&t_)
			{
			}
			const TableBase &get_table() const { return *_t; }
		};

	template <
		typename TableBase
		, typename LockOwner
		, typename LockContent
	>
		auto make_bucket_print(const TableBase &t_, LockOwner &lk_, LockContent &co_)
		{
			return bucket_print<TableBase, LockOwner, LockContent>(t_, lk_, co_);
		}

	template <
		typename TableBase
		, typename LockOwner
		, typename LockContent
	>
		auto operator<<(
			std::ostream &o
			, const impl::bucket_print<TableBase, LockOwner, LockContent> &
		) -> std::ostream &;

	template <
		typename TableBase
	>
		auto make_table_print(const TableBase &t_)
		{
			return table_print<TableBase>(t_);
		}

	template <
		typename TableBase
	>
		class table_dump
		{
			const TableBase *_t;
		public:
			table_dump(const TableBase &t_)
				: _t(&t_)
			{}
			const TableBase &get_table() const { return *_t; }
		};

	template <
		typename TableBase
	>
		auto make_table_dump(const TableBase &t_)
		{
			return table_dump<TableBase>(t_);
		}

	template <
		typename TableBase
	>
		auto operator<<(
			std::ostream &o
			, const impl::table_dump<TableBase> &
		) -> std::ostream &;

	template <
		typename TableBase
		, typename LockOwner
		, typename LockContent
	>
		auto operator<<(
			std::ostream &
			, const bucket_print<TableBase, LockOwner, LockContent> &
		) -> std::ostream &;

	template <
		typename TableBase
	>
		auto operator<<(
			std::ostream &o
			, const impl::table_print<TableBase> &t
		) -> std::ostream &;
}

#endif
