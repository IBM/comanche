#ifndef _DAWN_HOP_HASH_H
#define _DAWN_HOP_HASH_H

#include "segment_layout.h"

#include "trace_flags.h"
#include "persister.h"
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

	template <typename Referent>
		class bucket_ref
		{
			Referent *_ref;
			segment_and_bucket _sb;
		public:
			bucket_ref(Referent *ref_, const segment_and_bucket &sb_)
				: _ref(ref_)
				, _sb(sb_)
			{
			}
			std::size_t index() const { return sb().index(); }
			const segment_and_bucket &sb() const { return _sb; }
			Referent &ref() const { return *_ref; }
		};

	template <typename Owner, typename SharedMutex>
		struct bucket_unique_lock
			: public bucket_ref<Owner>
			, public std::unique_lock<SharedMutex>
		{
			bucket_unique_lock(
				Owner &b_
				, const segment_and_bucket &i_
				, SharedMutex &m_
			)
				: bucket_ref<Owner>(&b_, i_)
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
				bucket_ref<Owner>::operator=(std::move(other_));
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

	template <typename BucketAligned, typename SharedMutex>
		struct bucket_shared_lock
			: public bucket_ref<BucketAligned>
			, public std::shared_lock<SharedMutex>
		{
			bucket_shared_lock(
				BucketAligned &b_
				, const segment_and_bucket &i_
				, SharedMutex &m_
			)
				: bucket_ref<BucketAligned>(&b_, i_)
				, std::shared_lock<SharedMutex>(m_)
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
				std::shared_lock<SharedMutex>::operator=(std::move(other_));
				bucket_ref<BucketAligned>::operator=(std::move(other_));
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
		{
			using bucket_aligned_t = bucket_aligned<Bucket>;
			bucket_mutexes<Mutex> *_bucket_mutexes;
			bucket_aligned_t *_b;
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
			using owner_unique_lock_t = bucket_unique_lock<owner, SharedMutex>;
			using owner_shared_lock_t = bucket_shared_lock<owner, SharedMutex>;
			using content_unique_lock_t = bucket_unique_lock<content_t, SharedMutex>;
			using content_shared_lock_t = bucket_shared_lock<content_t, SharedMutex>;
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
				return persist_controller_t::segment_count_actual();
			}

			auto bucket_ix(const hash_result_t h) const -> bix_t;
			auto bucket_expanded_ix(const hash_result_t h) const -> bix_t;

			auto nearest_free_bucket(segment_and_bucket bi) -> content_unique_lock_t;

			auto make_space_for_insert(
				bix_t bi
				, content_unique_lock_t bf
			) -> content_unique_lock_t;

			template <typename Lock>
				auto locate_key(
					Lock &bi
					, const key_type &k
				) const -> std::tuple<bucket_t *, segment_and_bucket>;

			void resize();
			void resize_pass1();
			void resize_pass2();
			auto locate_bucket_mutexes(
				const segment_and_bucket &
			) const -> bucket_mutexes_t &;

			auto make_owner_unique_lock(
				const segment_and_bucket &a
			) const -> owner_unique_lock_t;

			/* lock an owner which precedes content */
			auto make_owner_unique_lock(
				const content_unique_lock_t &
				, unsigned bkwd
			) const -> owner_unique_lock_t;

			auto make_owner_shared_lock(const key_type &k) const -> owner_shared_lock_t;
			auto make_owner_shared_lock(
				const segment_and_bucket &
			) const -> owner_shared_lock_t;

			auto make_content_unique_lock(
				const segment_and_bucket &
			) const -> content_unique_lock_t;
			/* lock content which follows an owner */
			auto make_content_unique_lock(
				const owner_unique_lock_t &
				, unsigned fwd
			) const -> content_unique_lock_t;

			using persist_controller_t::mask;

			auto owner_value_at(owner_unique_lock_t &bi) const -> owner::value_type;
			auto owner_value_at(owner_shared_lock_t &bi) const -> owner::value_type;

			auto make_segment_and_bucket(bix_t ix) const -> segment_and_bucket;
			auto make_segment_and_bucket_for_iterator(
				bix_t ix
			) const -> segment_and_bucket;
			auto make_segment_and_bucket_prev(
				segment_and_bucket a
				, unsigned bkwd
			) const -> segment_and_bucket;
			auto locate(const segment_and_bucket &) const -> bucket_aligned_t &;

			auto locate_owner(const segment_and_bucket &a) const -> const owner &;

			auto locate_content(const segment_and_bucket &a) const -> const content_t &;
			auto locate_content(const segment_and_bucket &a) -> content_t &;

			auto bucket(const key_type &) const -> size_type;
			auto bucket_size(const size_type n) const -> size_type;

			bool is_free_by_owner(const segment_and_bucket &a) const;
			bool is_free(const segment_and_bucket &a);
			bool is_free(const segment_and_bucket &a) const;

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
				, const Allocator &av = Allocator()
			);
			~table_base();
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
				return iterator(*this, make_segment_and_bucket(0U));
			}
			auto end() -> iterator
			{
				return iterator(
					*this
					, make_segment_and_bucket_for_iterator(bucket_count())
				);
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
				return const_iterator(*this, make_segment_and_bucket(0U));
			}
			auto cend() const -> const_iterator
			{
				return const_iterator(
					*this
					, make_segment_and_bucket_for_iterator(bucket_count())
				);
			}

			using persist_controller_t::bucket_count;
			using persist_controller_t::max_bucket_count;

			auto begin(size_type n) -> local_iterator
			{
				auto sb = make_segment_and_bucket(n);
				auto owner_lk = make_owner_shared_lock(sb);
				return local_iterator(*this, sb, locate_owner(sb).value(owner_lk));
			}
			auto end(size_type n) -> local_iterator
			{
				auto sb = make_segment_and_bucket(n);
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
				auto sb = make_segment_and_bucket(n);
				auto owner_lk = make_owner_shared_lock(sb);
				return const_local_iterator(*this, sb, locate_owner(sb).value(owner_lk));
			}
			auto cend(size_type n) const -> const_local_iterator
			{
				auto sb = make_segment_and_bucket(n);
				return const_local_iterator(*this, sb, owner::value_type(0));
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
			template <
				typename TableBase
				, typename Lock
			>
				friend
					auto operator<<(
						std::ostream &o_
						, const impl::owner_print<
							TableBase
							, Lock
						> &
					) -> std::ostream &;

			template <typename TableBase>
				friend
					auto operator<<(
						std::ostream &o_
						, const owner_print<
							TableBase
							, bypass_lock<const owner>
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
							TableBase
							, LockOwner
							, LockContent
						> &
					) -> std::ostream &;

			template <typename Table>
				friend
				auto operator<<(
					std::ostream &o_
					, const impl::bucket_print
					<
						table_base
						, impl::bypass_lock<const impl::owner>
						, impl::bypass_lock<
							const impl::content<typename Table::value_type>
						>
					> &
				) -> std::ostream &;
#endif
			template <typename Table>
				friend class impl::table_local_iterator_impl;
			template <typename Table>
				friend class impl::table_iterator_impl;
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
		using base = impl::table_base<Key, T, Hash, Pred, Allocator, SharedMutex>;
	public:
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
			persist_data_t *pc
			, const Allocator &av = Allocator()
		)
			: base(pc, av)
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

template <typename Table>
	bool operator!=(
		const impl::table_local_iterator_impl<Table> a
		, const impl::table_local_iterator_impl<Table> b
	);

template <typename Table>
	bool operator==(
		const impl::table_local_iterator_impl<Table> a
		, const impl::table_local_iterator_impl<Table> b
	);

template <typename Table>
	bool operator!=(
		const impl::table_iterator_impl<Table> a
		, const impl::table_iterator_impl<Table> b
	);

template <typename Table>
	bool operator==(
		const impl::table_iterator_impl<Table> a
		, const impl::table_iterator_impl<Table> b
	);

namespace impl
{
	template <typename Table>
		class table_local_iterator_impl
			: public std::iterator
				<
					std::forward_iterator_tag
					, typename Table::value_type
				>
		{
			const Table *_t;
			segment_and_bucket _sb;
			owner::value_type _mask;
			void advance_to_in_use()
			{
				if ( _mask != 0 )
				{
					for ( ; ( _mask & 1U ) == 0; _mask >>= 1 )
					{
						_sb.incr(*_t);
					}
				}
			}

		protected:
			using base =
				std::iterator<std::forward_iterator_tag, typename Table::value_type>;

			/* Table 106 (Iterator) */
			auto deref() const -> typename base::reference
			{
				return _t->locate(_sb).content<typename Table::value_type>::value();
			}
			void incr()
			{
				_sb.incr(*_t);
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
			table_local_iterator_impl(const Table &t_, const segment_and_bucket &sb_, owner::value_type mask_)
				: _t(&t_)
				, _sb(sb_)
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
			const Table *_t;
			segment_and_bucket _sb;
			void advance_to_in_use()
			{
				while ( _sb.index() < _t->bucket_count() && _t->is_free(_sb) )
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
				return _t->locate(_sb).content<typename Table::value_type>::value();
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
			table_iterator_impl(const Table &t_, const segment_and_bucket &sb_)
				: _t(&t_)
				, _sb(sb_)
			{
				advance_to_in_use();
			}
		};

	template <typename Table>
		class table_local_iterator
			: public table_local_iterator_impl<Table>
		{
			using typename table_local_iterator_impl<Table>::base;
		public:
			table_local_iterator(const Table &t_, const segment_and_bucket & sb_, owner::value_type mask_)
				: table_local_iterator_impl<Table>(t_, sb_, mask_)
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
			auto operator->() -> typename base::pointer
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
			using typename table_local_iterator_impl<Table>::base;
		public:
			table_const_local_iterator(const Table &t_, const segment_and_bucket & sb_, owner::value_type mask_)
				: table_local_iterator_impl<Table>(t_, sb_, mask_)
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
			auto operator->() -> const typename base::pointer
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
			using typename table_iterator_impl<Table>::base;
		public:
			table_iterator(const Table &t, const segment_and_bucket & i)
				: table_iterator_impl<Table>(t, i)
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
			auto operator->() -> typename base::pointer
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
			using typename table_iterator_impl<Table>::base;
		public:
			table_const_iterator(const Table &t, typename Table::size_type i)
				: table_iterator_impl<Table>(t, i)
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
			auto operator->() -> const typename base::pointer
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
}

template <typename Table>
	bool operator==(
		const impl::table_local_iterator_impl<Table> a
		, const impl::table_local_iterator_impl<Table> b
	)
	{
		assert(a._t == b._t);
		return a._mask == b._mask;
	}

template <typename Table>
	bool operator!=(
		const impl::table_local_iterator_impl<Table> a
		, const impl::table_local_iterator_impl<Table> b
	)
	{
		return !(a==b);
	}

template <typename Table>
	bool operator==(
		const impl::table_iterator_impl<Table> a
		, const impl::table_iterator_impl<Table> b
	)
	{
		assert(a._t == b._t);
		return a._sb == b._sb;
	}

template <typename Table>
	bool operator!=(
		const impl::table_iterator_impl<Table> a
		, const impl::table_iterator_impl<Table> b
	)
	{
		return !(a==b);
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
