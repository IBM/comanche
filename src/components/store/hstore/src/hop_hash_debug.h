/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


#ifndef _COMANCHE_HSTORE_HOP_HASH_DEBUG_H
#define _COMANCHE_HSTORE_HOP_HASH_DEBUG_H

#include "segment_and_bucket.h"
#include <cstddef> /* size_t */
#include <iosfwd>

namespace impl
{
	template <typename Value>
		class content;
	template <typename Bucket>
		class segment_and_bucket;
}

namespace impl
{
	class owner;

	template <typename Bucket, typename Referent>
		class bucket_ref;

	template <typename Bucket, typename Referent>
		class bypass_lock
			: public bucket_ref<Bucket, Referent>
		{
		public:
			bypass_lock(Referent &b_, const segment_and_bucket<Bucket> &i_)
				: bucket_ref<Bucket, Referent>(&b_, i_)
			{}
		};

	template <typename Value>
		auto operator<<(
			std::ostream &o
			, const impl::content<Value> &c
		) -> std::ostream &;

	template <
		typename Lock
	>
		class owner_print
		{
			std::size_t _ct;
			Lock *_i;
		public:
			owner_print(std::size_t ct_, Lock &i_)
				: _ct(ct_)
				, _i{&i_}
			{}
			std::size_t bucket_count() const { return _ct; }
			Lock &lock() const { return *_i; }
			auto sb() const { return lock().sb(); }
			std::size_t index() const { return lock().index(); }
		};

	template <
		typename Lock
	>
		auto operator<<(
			std::ostream &o
			, const owner_print<Lock> &
		) -> std::ostream &;

	template <typename TableBase>
		auto operator<<(
			std::ostream &o
			, const owner_print<bypass_lock<typename TableBase::bucket_t, const owner>> &
		) -> std::ostream &;

	template <
		typename LockOwner
		, typename LockContent
	>
		class bucket_print
		{
			std::size_t _ct;
			LockOwner *_c;
			LockContent *_i;
		public:
			bucket_print(std::size_t bucket_count_, LockOwner &w, LockContent &i);
			std::size_t bucket_count() const { return _ct; }
			LockOwner &lock_owner() const { return *_c; }
			LockContent &lock() const { return *_i; }
			std::size_t index() const { return lock().index(); }
			auto sb() const { return lock().sb(); }
		};

	template <
		typename TableBase
	>
		class hop_hash_print
		{
			const TableBase *_t;
		public:
			hop_hash_print(const TableBase &t_)
				: _t(&t_)
			{
			}
			const TableBase &get_hop_hash() const { return *_t; }
		};

	template <
		typename LockOwner
		, typename LockContent
	>
		auto make_bucket_print(std::size_t ct_, LockOwner &lk_, LockContent &co_)
		{
			return bucket_print<LockOwner, LockContent>(ct_, lk_, co_);
		}

	template <
		typename LockOwner
		, typename LockContent
	>
		auto operator<<(
			std::ostream &o
			, const impl::bucket_print<LockOwner, LockContent> &
		) -> std::ostream &;

	template <
		typename TableBase
	>
		auto make_hop_hash_print(const TableBase &t_)
		{
			return hop_hash_print<TableBase>(t_);
		}


	template <bool>
		class dump;

	template<>
		class dump<true>
		{
		public:
			template <
				typename TableBase
			>
				class hop_hash_dump
				{
					const TableBase *_t;
				public:
					hop_hash_dump(const TableBase &t_)
						: _t(&t_)
					{}
					const TableBase &get_hop_hash() const { return *_t; }
				};

			template <
				typename TableBase
			>
				static hop_hash_dump<TableBase> make_hop_hash_dump(const TableBase &t_)
				{
					return hop_hash_dump<TableBase>(t_);
				}

			template <
				typename Lock
			>
				static auto make_owner_print(
					const std::size_t &sz_
					, Lock &lk_
				) -> owner_print<Lock>
				{
					return owner_print<Lock>(sz_, lk_);
				}
		};

	template<>
		class dump<false>
		{
			template <
				typename TableBase
			>
				class hop_hash_dump
				{
					const TableBase *_t;
				public:
					hop_hash_dump(const TableBase &t_)
						: _t(&t_)
					{}
					const TableBase &get_hop_hash() const { return *_t; }
				};
		public:
			template <
				typename TableBase
			>
				static hop_hash_dump<TableBase> make_hop_hash_dump(const TableBase &t_)
				{
					return hop_hash_dump<TableBase>(t_);
				}

			template <
				typename Lock
			>
				static auto make_owner_print(
					const std::size_t &sz_
					, Lock &lk_
				) -> owner_print<Lock>
				{
					return owner_print<Lock>(sz_, lk_);
				}
		};

	template <
		typename TableBase
	>
		auto operator<<(
			std::ostream &o
			, const impl::dump<false>::hop_hash_dump<TableBase> &
		) -> std::ostream &;

	template <
		typename TableBase
	>
		auto operator<<(
			std::ostream &o
			, const impl::dump<true>::hop_hash_dump<TableBase> &
		) -> std::ostream &;

	template <
		typename LockOwner
		, typename LockContent
	>
		auto operator<<(
			std::ostream &
			, const bucket_print<LockOwner, LockContent> &
		) -> std::ostream &;

	template <
		typename TableBase
	>
		auto operator<<(
			std::ostream &o
			, const impl::hop_hash_print<TableBase> &t
		) -> std::ostream &;
}

#endif
