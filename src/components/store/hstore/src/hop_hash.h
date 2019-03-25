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


#ifndef _COMANCHE_HSTORE_HOP_HASH_H
#define _COMANCHE_HSTORE_HOP_HASH_H

#include "segment_layout.h"

#include "bucket_control_unlocked.h"
#include "construction_mode.h"
#include "trace_flags.h"
#include "persist_controller.h"
#include "segment_and_bucket.h"

#include <boost/iterator/transform_iterator.hpp>

#include <atomic>
#include <cassert>
#include <cstddef> /* size_t */
#include <cstdint> /* uint32_t */
#include <functional> /* equal_to */
#include <limits>
#include <new> /* allocator */
#include <shared_mutex> /* shared_timed_mutex */
#include <stdexcept>
#include <string>
#include <utility> /* hash, pair */

/* Inteded to implement Hopscotch hashing
 * http://mcg.cs.tau.ac.il/papers/disc2008-hopscotch.pdf
 */

#if TRACE_CONTENT || TRACE_OWNER || TRACE_BUCKET
#include "hop_hash_debug.h"
#endif

#if TRACE_TABLE
template <
	typename Key
	, typename T
	, typename Hash
	, typename Pred
	, typename Allocator
	, typename SharedMutex
>
	class table;
#endif

namespace impl
{
	class no_near_empty_bucket
		: public std::range_error
	{
	public:
		using bix_t = std::size_t;
	private:
		bix_t _bi;
	public:
		no_near_empty_bucket(bix_t bi, std::size_t size, const std::string &why);
		bix_t bi() const { return _bi; }
	};

	class move_stuck
		: public no_near_empty_bucket
	{
	public:
		move_stuck(bix_t bi, std::size_t size);
	};

	class table_full
		: public no_near_empty_bucket
	{
	public:
		table_full(bix_t bi, std::size_t size);
	};

	template <typename Table>
		class table_iterator;

	template <typename Table>
		class table_const_iterator;

	template <typename Table>
		class table_local_iterator;

	template <typename Table>
		class table_const_local_iterator;

	template <typename Bucket, typename Referent>
		class bucket_ref
		{
		public:
			using segment_and_bucket_t = segment_and_bucket<Bucket>;
		private:
			Referent *_ref;
			segment_and_bucket_t _sb;
		public:
			bucket_ref(Referent *ref_, const segment_and_bucket_t &sb_)
				: _ref(ref_)
				, _sb(sb_)
			{
			}
			std::size_t index() const { return sb().index(); }
			const segment_and_bucket_t &sb() const { return _sb; }
			Referent &ref() const { return *_ref; }
		};

	template <typename Bucket, typename Owner, typename SharedMutex>
		struct bucket_unique_lock
			: public bucket_ref<Bucket, Owner>
			, public std::unique_lock<SharedMutex>
		{
			using base_ref = bucket_ref<Bucket, Owner>;
			using segment_and_bucket_t = typename base_ref::segment_and_bucket_t;
			bucket_unique_lock(
				Owner &b_
				, const segment_and_bucket_t &i_
				, SharedMutex &m_
			)
				: bucket_ref<Bucket, Owner>(&b_, i_)
				, std::unique_lock<SharedMutex>(m_)
			{
#if TRACE_LOCK
				std::cerr << __func__ << " " << this->index() << "\n";
#endif
			}
			bucket_unique_lock(bucket_unique_lock &&) = default;
			bucket_unique_lock &operator=(bucket_unique_lock &&other_)
			{
#if TRACE_LOCK
				std::cerr << __func__ << " " << this->index()
					<< "->" << other_.index() << "\n";
#endif
				std::unique_lock<SharedMutex>::operator=(std::move(other_));
				base_ref::operator=(std::move(other_));
				return *this;
			}
			~bucket_unique_lock()
			{
#if TRACE_LOCK
				if ( this->owns_lock() )
				{
					std::cerr << __func__ << " " << this->index() << "\n";
				}
#endif
			}
			template <typename Table>
				void assert_clear(bool b, Table &t)
				{
					this->ref().assert_clear(b, *this, t);
				}
		};

	template <typename Bucket, typename Referent, typename SharedMutex>
		struct bucket_shared_lock
			: public bucket_ref<Bucket, Referent>
			, public std::shared_lock<SharedMutex>
		{
			using base_ref = bucket_ref<Bucket, Referent>;
			using base_lock = std::shared_lock<SharedMutex>;
			using segment_and_bucket_t = typename base_ref::segment_and_bucket_t;
			bucket_shared_lock(
				Bucket &b_
				, const segment_and_bucket_t &i_
				, SharedMutex &m_
			)
				: base_ref(&b_, i_)
				, base_lock(m_)
			{
#if TRACE_LOCK
				std::cerr << __func__ << " " << this->index() << "\n";
#endif
			}
			bucket_shared_lock(bucket_shared_lock &&) = default;
			auto operator=(bucket_shared_lock &&other_) -> bucket_shared_lock &
			{
#if TRACE_LOCK
				std::cerr << __func__ << " " << this->index()
					<< "->" << other_.index() << "\n";
#endif
				base_lock::operator=(std::move(other_));
				base_ref::operator=(std::move(other_));
				return *this;
			}
			~bucket_shared_lock()
			{
#if TRACE_LOCK
				if ( this->owns_lock() )
				{
					std::cerr << __func__ << " " << this->index() << "\n";
				}
#endif
			}
		};

	template <typename Mutex>
		struct bucket_mutexes
		{
			Mutex _m_owner;
			Mutex _m_content;
			/* current state of ownership *for table::lock_shared/lock_uniqe/unlock
			 * purposes only*. Not maintained (or needed) for other users.
			 */
			enum { SHARED, UNIQUE } _state;
			bucket_mutexes()
				: _m_owner{}
				, _m_content{}
				, _state{SHARED}
			{}
		};

	template <typename Bucket, typename Mutex>
		struct bucket_control
			: public bucket_control_unlocked<Bucket>
		{
			bucket_mutexes<Mutex> *_bucket_mutexes;
		public:
			using base = bucket_control_unlocked<Bucket>;
			using bucket_aligned_t = typename base::bucket_aligned_t;
			using six_t = typename base::six_t;
			explicit bucket_control(
				six_t index_
				, bucket_aligned_t *buckets_
			)
				: bucket_control_unlocked<Bucket>(index_, buckets_)
				, _bucket_mutexes(nullptr)
			{}
			explicit bucket_control()
				: bucket_control(0U, nullptr)
			{}
		};

	template <typename Allocator>
		class table_allocator
			: public Allocator
		{
		public:
			explicit table_allocator(const Allocator &av_)
				: Allocator(av_)
			{}
		};

	template <typename Table>
		class table_local_iterator_impl;

	template <typename Table>
		class table_iterator_impl;

	template <
		typename Key
		, typename T
		, typename Hash
		, typename Pred
		, typename Allocator
		, typename SharedMutex
	>
		class table_base
			: private table_allocator<Allocator>
			, private segment_layout
			, private
				persist_controller<
					typename Allocator::template rebind<
						std::pair<const Key, T>
					>::other
				>
		{
		public:
			using key_type        = Key;
			using mapped_type     = T;
			using value_type      = std::pair<const key_type, mapped_type>;
			using persist_controller_t =
				persist_controller<
					typename Allocator::template rebind<value_type>::other
				>;
			using size_type       = typename persist_controller_t::size_type;
			using hasher          = Hash;
			using key_equal       = Pred;
			using allocator_type  = Allocator;
			using pointer         = typename allocator_type::pointer;
			using const_pointer   = typename allocator_type::const_pointer;
			using reference       = typename allocator_type::reference;
			using const_reference = typename allocator_type::const_reference;
			using iterator        = table_iterator<table_base>;
			using const_iterator  = table_const_iterator<table_base>;
			using local_iterator  = table_local_iterator<table_base>;
			using const_local_iterator = table_const_local_iterator<table_base>;
			using persist_data_t =
				persist_map<typename Allocator::template rebind<value_type>::other>;
		private:
			using bix_t = size_type; /* sufficient for all bucket indexes */
			using hash_result_t = typename hasher::result_type;
			using bucket_t = hash_bucket<value_type>;
			using content_t = content<value_type>;
			using bucket_mutexes_t = bucket_mutexes<SharedMutex>;
			using bucket_control_t = bucket_control<bucket_t, SharedMutex>;
			using bucket_aligned_t = typename bucket_control_t::bucket_aligned_t;
			using bucket_allocator_t =
				typename Allocator::template rebind<bucket_aligned_t>::other;
			using owner_unique_lock_t = bucket_unique_lock<bucket_t, owner, SharedMutex>;
			using owner_shared_lock_t = bucket_shared_lock<bucket_t, owner, SharedMutex>;
			using content_unique_lock_t = bucket_unique_lock<bucket_t, content_t, SharedMutex>;
			using content_shared_lock_t = bucket_shared_lock<bucket_t, content_t, SharedMutex>;
			using segment_and_bucket_t = segment_and_bucket<bucket_t>;
			static constexpr auto _segment_capacity =
				persist_controller_t::_segment_capacity;
			/* Need to adjust hash and bucket_ix interpretations in more places
			* before this can becom non-zero
			*/
			static constexpr unsigned log2_base_segment_size =
				persist_controller_t::log2_base_segment_size;
			static constexpr bix_t base_segment_size =
				persist_controller_t::base_segment_size;
			static_assert(
				owner::size < base_segment_size
				, "Base segment size must exceeed owner size"
			);
			using six_t = std::size_t; /* segment indexes (but uint8_t would do) */

			/* The number of non-null elements of _b may exceed _count
			* only if a crash occurred between the assignemnt of a non-null
			* value to _b[_count] and the increment of _count.
			* In that case the non-null value at _count points to lost memory.
			* That memory can be recovered the next time _b is extended.
			*/
			hasher _hasher;

			bucket_control_t _bc[_segment_capacity];

			six_t segment_count() const override
			{
				return persist_controller_t::segment_count_actual().value();
			}

			auto bucket_ix(const hash_result_t h) const -> bix_t;
			auto bucket_expanded_ix(const hash_result_t h) const -> bix_t;

			auto nearest_free_bucket(segment_and_bucket_t bi) -> content_unique_lock_t;

			auto make_space_for_insert(
				bix_t bi
				, content_unique_lock_t bf
			) -> content_unique_lock_t;

			template <typename Lock>
				auto locate_key(
					Lock &bi
					, const key_type &k
				) const -> std::tuple<bucket_t *, segment_and_bucket_t>;

			void resize();
			void resize_pass1();
			void resize_pass2();
			void resize_pass2_inner(
				bix_t ix_senior
				, bucket_control_t &junior_bucket_control
				, content_unique_lock_t &populated_content_lk
			);
			auto locate_bucket_mutexes(
				const segment_and_bucket_t &
			) const -> bucket_mutexes_t &;

			auto make_owner_unique_lock(
				const segment_and_bucket_t &a
			) const -> owner_unique_lock_t;

			/* lock an owner which precedes content */
			auto make_owner_unique_lock(
				const content_unique_lock_t &
				, unsigned bkwd
			) const -> owner_unique_lock_t;

			auto make_owner_shared_lock(const key_type &k) const -> owner_shared_lock_t;
			auto make_owner_shared_lock(
				const segment_and_bucket_t &
			) const -> owner_shared_lock_t;

			auto make_content_unique_lock(
				const segment_and_bucket_t &
			) const -> content_unique_lock_t;
			/* lock content which follows an owner */
			auto make_content_unique_lock(
				const owner_unique_lock_t &
				, unsigned fwd
			) const -> content_unique_lock_t;

			using persist_controller_t::mask;

			auto owner_value_at(owner_unique_lock_t &bi) const -> owner::value_type;
			auto owner_value_at(owner_shared_lock_t &bi) const -> owner::value_type;

			auto make_segment_and_bucket(bix_t ix) const -> segment_and_bucket_t;
			auto make_segment_and_bucket_for_iterator(
				bix_t ix
			) const -> segment_and_bucket_t;
			auto make_segment_and_bucket_at_begin() const -> segment_and_bucket_t;
			auto make_segment_and_bucket_at_end() const -> segment_and_bucket_t;
			auto make_segment_and_bucket_prev(
				segment_and_bucket_t a
				, unsigned bkwd
			) const -> segment_and_bucket_t;

			static auto locate_owner(const segment_and_bucket_t &a) -> const owner &;

			auto bucket(const key_type &) const -> size_type;
			auto bucket_size(const size_type n) const -> size_type;

			auto owned_by_owner_mask(const segment_and_bucket_t &a) const -> owner::value_type;
			bool is_free_by_owner(const segment_and_bucket_t &a) const;
			bool is_free(const segment_and_bucket_t &a);
			bool is_free(const segment_and_bucket_t &a) const;

			/* computed distance from first to last, accounting for the possibility that
			 * last is smaller than first due to wrapping.
			 */
			auto distance_wrapped(bix_t first, bix_t last) -> unsigned;

			unsigned _locate_key_call;
			unsigned _locate_key_owned;
			unsigned _locate_key_unowned;
			unsigned _locate_key_match;
			unsigned _locate_key_mismatch;

		public:
			explicit table_base(
				persist_data_t *pc
				, construction_mode mode
				, const Allocator &av = Allocator()
			);
		protected:
			virtual ~table_base();
		public:
			table_base(const table_base &) = delete;
			table_base &operator=(const table_base &) = delete;
			allocator_type get_allocator() const noexcept
			{
				return static_cast<const table_allocator<Allocator> &>(*this);
			}

			template <typename ... Args>
				auto emplace(Args && ... args) -> std::pair<iterator, bool>;
			auto insert(const value_type &value) -> std::pair<iterator, bool>;
			auto erase(const key_type &key) -> size_type;
			auto at(const key_type &key) -> mapped_type &;
			auto at(const key_type &key) const -> const mapped_type &;
			auto count(const key_type &k) const -> size_type;
			auto begin() -> iterator
			{
				return iterator(make_segment_and_bucket_at_begin());
			}
			auto end() -> iterator
			{
				return iterator(make_segment_and_bucket_at_end());
			}
			auto begin() const -> const_iterator
			{
				return cbegin();
			}
			auto end() const -> const_iterator
			{
				return cend();
			}
			auto cbegin() const -> const_iterator
			{
				return const_iterator(make_segment_and_bucket_at_begin());
			}
			auto cend() const -> const_iterator
			{
				return const_iterator(make_segment_and_bucket_at_end());
			}

			using persist_controller_t::bucket_count;
			using persist_controller_t::max_bucket_count;

			auto begin(size_type n) -> local_iterator
			{
				auto sb = make_segment_and_bucket_for_iterator(n);
				auto owner_lk = make_owner_shared_lock(sb);
				return local_iterator(*this, sb, locate_owner(sb).value(owner_lk));
			}
			auto end(size_type n) -> local_iterator
			{
				auto sb = make_segment_and_bucket_for_iterator(n);
				return local_iterator(*this, sb, owner::value_type(0));
			}
			auto begin(size_type n) const -> const_local_iterator
			{
				return cbegin(n);
			}
			auto end(size_type n) const -> const_local_iterator
			{
				return cend(n);
			}
			auto cbegin(size_type n) const -> const_local_iterator
			{
				auto sb = make_segment_and_bucket_for_iterator(n);
				auto owner_lk = make_owner_shared_lock(sb);
				return const_local_iterator(sb, locate_owner(sb).value(owner_lk));
			}
			auto cend(size_type n) const -> const_local_iterator
			{
				auto sb = make_segment_and_bucket_for_iterator(n);
				return const_local_iterator(sb, owner::value_type(0));
			}

			/* use trylock to attempt a shared lock */
			auto lock_shared(const key_type &k) -> bool;
			/* use trylock to attempt a unique lock */
			auto lock_unique(const key_type &k) -> bool;
			/* unlock (from write if write exists, else read */
			void unlock(const key_type &key);
			auto size() const -> size_type;

#if TRACE_TABLE
			friend
				auto operator<< <>(
					std::ostream &
					, const impl::table_print<
						table<Key, T, Hash, Pred, Allocator, SharedMutex>
					> &
				) -> std::ostream &;

			friend
				auto operator<< <>(
					std::ostream &
					, const impl::table_dump<table_base> &
				) -> std::ostream &;
#endif
#if TRACE_OWNER
			template <typename Lock>
				friend
					auto operator<<(
						std::ostream &o_
						, const impl::owner_print<Lock> &
					) -> std::ostream &;

			template <typename TableBase>
				friend
					auto operator<<(
						std::ostream &o_
						, const owner_print<
							bypass_lock<typename TableBase::bucket_t, const owner>
						> &
					) -> std::ostream &;
#endif
#if TRACE_BUCKET
			template<
				typename TableBase
				, typename LockOwner
				, typename LockContent
			>
				friend
					auto operator<<(
						std::ostream &o_
						, const impl::bucket_print<
							LockOwner
							, LockContent
						> &
					) -> std::ostream &;

			template <typename Table>
				friend
				auto operator<<(
					std::ostream &o_
					, const bucket_print
					<
						bypass_lock<typename Table::bucket_t, const owner>
						, bypass_lock<
							typename Table::bucket_t
							, const content<typename Table::value_type>
						>
					> &
				) -> std::ostream &;
#endif
			template <typename Table>
				friend class impl::table_local_iterator_impl;
			template <typename Table>
				friend class impl::table_iterator_impl;
			template <typename Table>
				friend class impl::table_local_iterator;
			template <typename Table>
				friend class impl::table_const_local_iterator;
			template <typename Table>
				friend class impl::table_iterator;
			template <typename Table>
				friend class impl::table_const_iterator;
		};
}

template <
	typename Key
	, typename T
	, typename Hash = std::hash<Key>
	, typename Pred = std::equal_to<Key>
	, typename Allocator = std::allocator<std::pair<const Key, T>>
	, typename SharedMutex = std::shared_timed_mutex
>
	class table
		: private impl::table_base<Key, T, Hash, Pred, Allocator, SharedMutex>
	{
	public:
		using base = impl::table_base<Key, T, Hash, Pred, Allocator, SharedMutex>;
		using size_type      = std::size_t;
		using key_type       = Key;
		using mapped_type    = T;
		using value_type     = std::pair<const key_type, mapped_type>;
		/* base is private. Can we use it to specify public types? Yes. */
		using iterator       = impl::table_iterator<base>;
		using const_iterator = impl::table_const_iterator<base>;
		using persist_data_t = typename base::persist_data_t;
		using allocator_type = typename base::allocator_type;

		/* contruct/destroy/copy */
		explicit table(
			persist_data_t *pc_
			, construction_mode mode_
			, const Allocator &av_ = Allocator()
		)
			: base(pc_, mode_, av_)
		{}

		using base::get_allocator;

		/* size and capacity */
		auto empty() const noexcept -> bool
		{
			return size() == 0;
		}
		using base::size;
		using base::bucket_count;
		auto max_size() const noexcept -> size_type
		{
			return (1U << (base::_segment_capacity-1U));
		}
		/* iterators */

		using base::begin;
		using base::end;
		using base::cbegin;
		using base::cend;

		/* modifiers */
		template <typename ... Args>
			auto emplace(Args && ... args) -> std::pair<iterator, bool>
			{
				return base::emplace(std::forward<Args>(args)...);

			}
		auto insert(const value_type &value) -> std::pair<iterator, bool>
		{
			return base::insert(value);
		}
		auto erase(const key_type &key) -> size_type
		{
			return base::erase(key);
		}

		/* lookup */
		auto at(const key_type &key) const -> const mapped_type &
		{
			return base::at(key);
		}

		auto at(const key_type &key) -> mapped_type &
		{
			return base::at(key);
		}

		auto count(const key_type &key) const -> size_type
		{
			return base::count(key);
		}

		/* locking */
		auto lock_shared(const key_type &k) -> bool
		{
			return base::lock_shared(k);
		}

		auto lock_unique(const key_type &k) -> bool
		{
			return base::lock_unique(k);
		}

		void unlock(const key_type &key)
		{
			return base::unlock(key);
		}

		template <typename Table>
			friend class impl::table_local_iterator_impl;

		template <typename Table>
			friend class impl::table_iterator_impl;
	};

namespace impl
{
	template <typename Table>
		bool operator!=(
			const table_local_iterator_impl<Table> a
			, const table_local_iterator_impl<Table> b
		);

	template <typename Table>
		bool operator==(
			const table_local_iterator_impl<Table> a
			, const table_local_iterator_impl<Table> b
		);

	template <typename Table>
		bool operator!=(
			const table_iterator_impl<Table> a
			, const table_iterator_impl<Table> b
		);

	template <typename Table>
		bool operator==(
			const table_iterator_impl<Table> a
			, const table_iterator_impl<Table> b
		);

	template <typename Table>
		class table_local_iterator_impl
			: public std::iterator
				<
					std::forward_iterator_tag
					, typename Table::value_type
				>
		{
			using segment_and_bucket_t = typename Table::segment_and_bucket_t;
			segment_and_bucket_t _sb;
			owner::value_type _mask;
			void advance_to_in_use()
			{
				if ( _mask != 0 )
				{
					for ( ; ( _mask & 1U ) == 0; _mask >>= 1 )
					{
						_sb.incr_with_wrap();
					}
				}
			}

		protected:
			using base =
				std::iterator<std::forward_iterator_tag, typename Table::value_type>;

			/* Table 106 (Iterator) */
			auto deref() const -> typename base::reference
			{
				return _sb.deref().content<typename Table::value_type>::value();
			}
			void incr()
			{
				_sb.incr_with_wrap();
				_mask >>= 1;
				advance_to_in_use();
			}
		public:
			/* Table 17 (EqualityComparable) */
			friend
				bool operator== <>(
					table_local_iterator_impl<Table> a
					, table_local_iterator_impl<Table> b
				);
			/* Table 107 (InputIterator) */
			friend
				bool operator!= <>(
					table_local_iterator_impl<Table> a
					, table_local_iterator_impl<Table> b
				);
			/* Table 109 (ForwardIterator) - handled by 107 */
		public:
			table_local_iterator_impl(const segment_and_bucket_t &sb_, owner::value_type mask_)
				: _sb(sb_)
				, _mask(mask_)
			{
				advance_to_in_use();
			}
		};

	template <typename Table>
		class table_iterator_impl
			: public std::iterator
				<
					std::forward_iterator_tag
					, typename Table::value_type
				>
		{
			using segment_and_bucket_t = typename Table::segment_and_bucket_t;
			segment_and_bucket_t _sb;
			void advance_to_in_use()
			{
				while ( _sb.can_incr_without_wrap() && _sb.deref().state_get() != Table::bucket_t::IN_USE )
				{
					_sb.incr_without_wrap();
				}
			}

		protected:
			using base =
				std::iterator<std::forward_iterator_tag, typename Table::value_type>;

			/* Table 106 (Iterator) */
			auto deref() const -> typename base::reference
			{
				return _sb.deref().content<typename Table::value_type>::value();
			}
			void incr()
			{
				_sb.incr_without_wrap();
				advance_to_in_use();
			}
		public:
			/* Table 17 (EqualityComparable) */
			friend
				bool operator== <>(
					table_iterator_impl<Table> a
					, table_iterator_impl<Table> b
				);
			/* Table 107 (InputIterator) */
			friend
				bool operator!= <>(
					table_iterator_impl<Table> a
					, table_iterator_impl<Table> b
				);
			/* Table 109 (ForwardIterator) - handled by 107 */
		public:
			table_iterator_impl(const segment_and_bucket_t &sb_)
				: _sb(sb_)
			{
				advance_to_in_use();
			}
		};

	template <typename Table>
		class table_local_iterator
			: public table_local_iterator_impl<Table>
		{
			using segment_and_bucket_t = typename Table::segment_and_bucket_t;
			using typename table_local_iterator_impl<Table>::base;
		public:
			table_local_iterator(const Table &t_, const segment_and_bucket_t & sb_, owner::value_type mask_)
				: table_local_iterator_impl<Table>(sb_, mask_)
			{}
			/* Table 106 (Iterator) */
			auto operator*() const -> typename base::reference
			{
				return this->deref();
			}
			auto operator++() -> table_local_iterator &
			{
				this->incr();
				return *this;
			}
			/* Table 17 (EqualityComparable) */
			/* Table 107 (InputIterator) */
			auto operator->() const -> typename base::pointer
			{
				return &this->deref();
			}
			auto operator++(int) -> table_local_iterator
			{
				auto ti = *this;
				this->incr();
				return ti;
			}
			/* Table 109 (ForwardIterator) - handled by 107 */
		};

	template <typename Table>
		class table_const_local_iterator
			: public table_local_iterator_impl<Table>
		{
			using segment_and_bucket_t = typename Table::segment_and_bucket_t;
			using typename table_local_iterator_impl<Table>::base;
		public:
			table_const_local_iterator(const segment_and_bucket_t & sb_, owner::value_type mask_)
				: table_local_iterator_impl<Table>(sb_, mask_)
			{}
			/* Table 106 (Iterator) */
			auto operator*() const -> const typename base::reference
			{
				return this->deref();
			}
			auto operator++() -> table_const_local_iterator &
			{
				this->incr();
				return *this;
			}
			/* Table 17 (EqualityComparable) */
			/* Table 107 (InputIterator) */
			auto operator->() const -> const typename base::pointer
			{
				return &this->deref();
			}
			auto operator++(int) -> table_const_local_iterator
			{
				auto ti = *this;
				this->incr();
				return ti;
			}
			/* Table 109 (ForwardIterator) - handled by 107 */
		};

	template <typename Table>
		class table_iterator
			: public table_iterator_impl<Table>
		{
			using segment_and_bucket_t = typename Table::segment_and_bucket_t;
			using typename table_iterator_impl<Table>::base;
		public:
			table_iterator(const segment_and_bucket_t & i)
				: table_iterator_impl<Table>(i)
			{}
			/* Table 106 (Iterator) */
			auto operator*() const -> typename base::reference
			{
				return this->deref();
			}
			auto operator++() -> table_iterator &
			{
				this->incr();
				return *this;
			}
			/* Table 17 (EqualityComparable) */
			/* Table 107 (InputIterator) */
			auto operator->() const -> typename base::pointer
			{
				return &this->deref();
			}
			table_iterator operator++(int)
			{
				auto ti = *this;
				this->incr();
				return ti;
			}
			/* Table 109 (ForwardIterator) - handled by 107 */
		};

	template <typename Table>
		class table_const_iterator
			: public table_iterator_impl<Table>
		{
			using segment_and_bucket_t = typename Table::segment_and_bucket_t;
			using typename table_iterator_impl<Table>::base;
		public:
			table_const_iterator(typename Table::size_type i)
				: table_iterator_impl<Table>(i)
			{}
			/* Table 106 (Iterator) */
			auto operator*() const -> const typename base::reference
			{
				return this->deref();
			}
			auto operator++() -> table_const_iterator &
			{
				this->incr();
				return *this;
			}
			/* Table 17 (EqualityComparable) */
			/* Table 107 (InputIterator) */
			auto operator->() const -> const typename base::pointer
			{
				return &this->deref();
			}
			table_const_iterator operator++(int)
			{
				auto ti = *this;
				this->incr();
				return ti;
			}
			/* Table 109 (ForwardIterator) - handled by 107 */
		};

	template <typename Table>
		bool operator==(
			const table_local_iterator_impl<Table> a
			, const table_local_iterator_impl<Table> b
		)
		{
			return a._mask == b._mask;
		}

	template <typename Table>
		bool operator!=(
			const table_local_iterator_impl<Table> a
			, const table_local_iterator_impl<Table> b
		)
		{
			return !(a==b);
		}

	template <typename Table>
		bool operator==(
			const table_iterator_impl<Table> a
			, const table_iterator_impl<Table> b
		)
		{
			return a._sb == b._sb;
		}

	template <typename Table>
		bool operator!=(
			const table_iterator_impl<Table> a
			, const table_iterator_impl<Table> b
		)
		{
			return !(a==b);
		}
}

#if TRACK_OWNER
#include "owner.tcc"
#endif
#include "persist_controller.tcc"
#include "table.tcc"

#if TRACE_CONTENT || TRACE_OWNER || TRACE_BUCKET
#include "hop_hash_debug.tcc"
#endif

#endif
