/*
 * Hopscotch hash table - template Key, Value, and allocators
 */

#include "perishable.h"
#include "perishable_expiry.h"
#include "persistent.h"
#include "pobj_allocator.h"

#include <boost/iterator/transform_iterator.hpp>

#include <algorithm>
#include <cassert>
#include <exception>
#if TRACE_PERISHABLE_EXPIRY
#include <iostream> /* for perishable_expiry */
#endif
#include <utility> /* move */

#if TRACE_MANY
#include <sstream> /* ostringstream */
#endif

/*
 * ===== table_base =====
 */

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::table_base(
		persist_data_t *pc_
		, const Alloc &av_
	)
		: Alloc{av_}
		, _hasher{}
		, _pc(av_, pc_)
		, _locate_key_call(0)
		, _locate_key_owned(0)
		, _locate_key_unowned(0)
		, _locate_key_match(0)
		, _locate_key_mismatch(0)
	{
		const auto bp_src = _pc.bp_src();
		const auto bc_dst =
			boost::make_transform_iterator(_bc, std::mem_fn(&bucket_control_t::_b));
		std::transform(
			bp_src
			, bp_src + _segment_capacity
			, bc_dst
			, [] (const auto &c) { return &*c; }
		);
		_bc[0]._bucket_mutexes = new bucket_mutexes_t[base_segment_size];
		for ( auto ix = 1U; ix != _pc.segment_count_actual(); ++ix )
		{
			_bc[ix]._bucket_mutexes = new bucket_mutexes_t[base_segment_size << (ix-1U)];
		}
#if TRACE_MANY
		std::cerr << __func__ << " count " << _pc.segment_count_actual()
			<< " count_target " << _pc.segment_count_target() << "\n";
#endif
		/* If in the middle of a resize op, rerun the resize. */
		if ( _pc.segment_count_actual() != _pc.segment_count_target() )
		{
			assert( _pc.segment_count_actual() + 1U == _pc.segment_count_target() );
			resize();
		}

		if ( _pc.is_size_unstable() )
		{
			size_type s = 0U;
			/* a slow way to find the table size, but we have no other way. */
			for ( const auto i : *this )
			{
				(void)i;
				++s;
			}
			_pc.size_set(s);
		}
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::~table_base()
	{
		perishable::report();
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::is_free_by_owner(
		const segment_and_bucket &a_
	) const -> bool
	{
		/* In order to develop a mask of "used" (non-free) locations,
		 * examine the members of every owner starting with the leftmost
		 * bucket which can include bi in its owner.
		 */
		owner::value_t c = 0U;
		auto sbw = make_segment_and_bucket_prev(a_, owner::size);
		for ( auto owner_lk = make_owner_unique_lock(sbw)
			; owner_lk.sb() != a_
			; sbw.incr(*this), owner_lk = make_owner_unique_lock(sbw)
		)
		{
			c >>= 1U;
			/*
			 * The "used" flag of a bucket is held by at most one owner.
			 * Therefore, no corresponding bits shall be 1 in both c
			 * (the sum of previous owners) and _b[owner_lk.index()]._owner
			 * (the current owner to be included).
			 */

			auto sbw2 = sbw;
			sbw2.incr(*this);
#if TRACE_PERISHABLE_EXPIRY
			if ( (c & locate_owner(sbw2).value(owner_lk)) != 0 )
			{
				std::cerr << __func__ << " ownership disagreement in range ["
					<< bucket_ix(a_.index()-owner::size) << ".." << sbw2.index()
					<< "]\n";
			}
#else
			assert( (c & locate_owner(sbw2).value(owner_lk)) == 0 );
#endif
			c |= locate_owner(sbw2).value(owner_lk);
		}
		/* c now contains, in its 0 bit, the "used" aspect of _b[bi_] */
		return ( c & 1U ) == 0U;
	}

/* NOTE: is_free is inherently non-const, except where checking program logic */
template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::is_free(
		const segment_and_bucket &a_
	) const -> bool
	{
		auto &b_src = locate(a_);
		switch ( b_src.state_get() )
		{
			case bucket_t::FREE:
				return true;
			case bucket_t::IN_USE:
				return false;
			default:
				return is_free_by_owner(a_);
		}
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::is_free(
		const segment_and_bucket &a_
	) -> bool
	{
		auto &b_src = locate_content(a_);
		switch ( b_src.state_get() )
		{
			case bucket_t::FREE:
				return true;
			case bucket_t::IN_USE:
				return false;
			default:
			{
				auto f = is_free_by_owner(a_);
				b_src.state_set(f ? bucket_t::FREE : bucket_t::IN_USE);
				return f;
			}
		}
	}

/* From a hash, return the index of the bucket (in seg_entry) to which it maps */
template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::bucket_ix(
		const hash_result_t h_
	) const -> bix_t
	{
#if DEBUG_TRACE_BUCKET_IX
		std::cerr << __func__ << " h_ " << h_
			<< " mask " << mask() << " -> " << ( h_ & mask() ) << "\n";
#endif
		return h_ & mask();
	}

/* From a hash, return the index of the bucket (in seg_entry) to which it maps,
 * including a single new segment not in count.
 */
template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::bucket_expanded_ix(
		const hash_result_t h_
	) const -> bix_t
	{
		auto mask_expanded = (mask() << 1U) + 1;
#if DEBUG_TRACE_BUCKET_IX
		std::cerr << __func__ << " h_ " << h_
			<< " mask " << mask_expanded << " -> " << ( h_ & mask_expanded ) << "\n";
#endif
		return h_ & mask_expanded;
	}

/*
 * Precondition: hold owner unique lock on bi.
 * Exit with content unique lock on the free bucket (while retaining owner unique lock on bi).
 */
template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::nearest_free_bucket(
		segment_and_bucket bi_
	) -> content_unique_lock_t
	{
		/* acquire content_lock(cursor) by virtue of owning owner_lock(cursor) */
		const auto start = bi_;
		auto content_lk = make_content_unique_lock(bi_);

		while ( ! is_free(content_lk.sb()) )
		{
			bi_.incr(*this);
			content_lk = make_content_unique_lock(bi_);
			if ( content_lk.sb() == start )
			{
				throw table_full{bi_.index(), bucket_count()};
			}
		}
#if TRACE_MANY
		std::cerr << __func__
			<< "(" << start.index() << ") => " << content_lk.index() << "\n";
#endif
		content_lk.assert_clear(true, *this);
		return content_lk;
	}

/* Precondition: holdss owner_unique_lock on bi_.
 *
 * Receives an owner index bi_, and a content unique lock on the free content at b_dst_lock_
 *
 * Returns a free content index within range of bi_, and a lock on that free content index
 */
template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::make_space_for_insert(
		bix_t bi_
		, content_unique_lock_t b_dst_lock_
	) -> content_unique_lock_t
	{
		auto free_distance =
			(b_dst_lock_.index() < bi_
			? b_dst_lock_.index() + bucket_count()
			: b_dst_lock_.index()) - bi_
			;
		while ( owner::size <= free_distance )
		{
#if TRACE_MANY
			std::cerr << __func__
				<< " owner size " << owner::size
				<< " < free distance " << free_distance
				<< "\n"
				<< make_table_dump(*this);
#endif
			/*
			 * Relocate an element in some possible owner of the free element, bf,
			 * to somewhere else in that owner's owner.
			 *
			 * The owners to check are, in order,
			 *   b_dst_lock_-owner::size-1 through b_dst_lock__-1.
			 */

			/* The bucket containing an item to be relocated */
			auto owner_lock = make_owner_unique_lock(b_dst_lock_, owner::size - 1U);
			/* Any item in bucket owner_lock precedes b_dst_lock_, and is eligible for move */
			auto eligible_items = ( 1U << owner::size ) - 1U;
			while (
				owner_lock.sb() != b_dst_lock_.sb()
				&&
				(owner_lock.ref().value(owner_lock) & eligible_items) == 0
			)
			{
				auto sb = owner_lock.sb();
				sb.incr(*this);
				owner_lock = make_owner_unique_lock(sb);
				/* The leftmost eligible item is no longer eligible; remove it */
				eligible_items >>= 1U;
			}

			/* postconditions
			 *   owner_lock.index() == b_dst_lock_ : we ran out of items to examine
			 *  or
			 *   (_b[lock.index()]._owner & eligible_items) != 0 : at least one item in owner
			 *     is eliglbe for the move; the best item is the 1 at the smallest index
			 *     in _b[lock.index()]._owner & eligible_items.
			 */
			if ( owner_lock.sb() == b_dst_lock_.sb() )
			{
				/* If no bucket was found which owned possible elements to move,
				 * we are stuck, no element can be moved.
				 */
				throw move_stuck(bi_, bucket_count());
			}

			const auto c = owner_lock.ref().value(owner_lock) & eligible_items;
			assert(c != 0);
			assert( c == unsigned(c) );

			/* find index of first (rightmost) one in c */
			const auto p = unsigned(__builtin_ctz(unsigned(c)));
			/* content to relocate */
			auto b_src_lock = make_content_unique_lock(owner_lock, p);

			b_src_lock.assert_clear(false, *this);
			b_dst_lock_.assert_clear(true, *this);
#if TRACK_OWNER
			b_src_lock.ref().owner_verify(owner_lock.index());
#endif
#if TRACE_MANY
			std::ostringstream c_old;
			{
				c_old << make_owner_print(*this, owner_lock);
			}
#endif
			b_dst_lock_.ref().content_share(b_src_lock.ref());
			/* The owner will
			 *  a) lose at element at position p and
			 *  b) gain the element at position b_dst_lock_ (relative to lock.index())
			 */
			const auto q =
				unsigned(
					(
						b_dst_lock_.index() < owner_lock.index()
						? b_dst_lock_.index() + bucket_count()
						: b_dst_lock_.index()
					)
					- owner_lock.index()
				)
				;
			/*
			 * ownership is moving from bf to owner
			 * 1. mark src EXITING and dst ENTERING
			 *   flush
			 * 2. update owner (atomic nove of in_use bit)
			 *   flush
			 * 3. mark src FREE and dst IN_USE
			 *   flush
			 */
			assert(!is_free(b_src_lock.sb()));
			assert(is_free(b_dst_lock_.sb()));
			b_src_lock.ref().state_set(bucket_t::EXITING);
			_pc.persist_content(b_src_lock.ref(), "content exiting");
			b_dst_lock_.ref().state_set(bucket_t::ENTERING);
			_pc.persist_content(b_dst_lock_.ref(), "context entering");

			owner_lock.ref().move(q, p, owner_lock);
			_pc.persist_owner(owner_lock.ref(), "owner update");

			b_src_lock.ref().erase();
			b_src_lock.assert_clear(true, *this);
			b_dst_lock_.assert_clear(false, *this);
			b_src_lock.ref().state_set(bucket_t::FREE);
			_pc.persist_content(b_src_lock.ref(), "content free");
			b_dst_lock_.ref().state_set(bucket_t::IN_USE);
			_pc.persist_content(b_dst_lock_.ref(), "context in_use");
			/* New free location */
#if TRACE_MANY
			std::cerr << __func__
				<< " bucket " << owner_lock.index()
				<< " move " << b_src_lock.index() << "->" << b_dst_lock_.index()
				<< " "
				<< c_old.str()
				<< "->"
				<< make_owner_print(*this, owner_lock)
				<< "\n";
#endif
			b_dst_lock_ = std::move(b_src_lock);

			free_distance =
				(
					b_dst_lock_.index() < bi_
					? b_dst_lock_.index() + bucket_count()
					: b_dst_lock_.index()
				)
				- bi_
				;
		}
		/* postcondition:
		 *   free_distance < owner::size
		 *
		 * Enter the new item in b_dst_lock_ and update the ownership at bi_
		 */
#if TRACE_MANY
		std::cerr << __func__ << " exit, free distance " << free_distance
			<< "\n" << make_table_dump(*this);
#endif
		b_dst_lock_.assert_clear(true, *this);
		return b_dst_lock_;
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	template <typename ... Args>
		auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::emplace(
			Args && ... args
		) -> std::pair<iterator, bool>
		try
		{
#if TRACE_MANY
			std::cerr << __func__ << " BEGIN LIST\n"
				<< make_table_dump(*this)
				<< __func__ << " END LIST\n";
#endif
		RETRY:
			auto v = value_type(std::forward<Args>(args)...);
			/* The bucket in which to place the new entry */
			auto sbw = make_segment_and_bucket(bucket(v.first));
			auto owner_lk = make_owner_unique_lock(sbw);

			/* If the key already exists, refuse to emplace */
			if ( auto cv = owner_lk.ref().value(owner_lk) )
			{
				auto sbc = sbw;
				for ( ; cv ; cv >>= 1U, sbc.incr(*this) )
				{
					if ( (cv & 1U) && key_equal()(locate(sbc).key(), v.first) )
					{
#if TRACE_MANY
						std::cerr << __func__ << " (already present)\n";
#endif
						return {iterator{*this, sbc}, false};
					}
				}
			}
			/* the nearest free bucket */
			try
			{
				auto b_dst = nearest_free_bucket(sbw);
				b_dst = make_space_for_insert(owner_lk.index(), std::move(b_dst));

				b_dst.assert_clear(true, *this);
				b_dst.ref().content_construct(owner_lk.index(), std::move(v));

				/* 3-step change to owner:
				 *  1. mark the content "ENTERING"
				 *   flush
				 *  2. atomically update the owner
				 *   flush
				 *  3. mark the content "IN_USE"
				 *   flush
				 */
				b_dst.ref().state_set(bucket_t::ENTERING);
				_pc.persist_content(b_dst.ref(), "content entering");
				_pc.size_destabilize();
				owner_lk.ref().insert(
					owner_lk.index()
					, unsigned(
						(
							b_dst.index() < owner_lk.index()
							? b_dst.index()+bucket_count()
							: b_dst.index()
						)
						- owner_lk.index()
					)
					, owner_lk
				);
				_pc.persist_owner(owner_lk.ref(), "owner emplace");
				b_dst.ref().state_set(bucket_t::IN_USE);
				_pc.persist_content(b_dst.ref(), "content in_use");
#if TRACE_MANY
				std::cerr << __func__ << " bucket " << owner_lk.index()
					<< " store at " << b_dst.index() << " "
					<< make_owner_print(*this, owner_lk)
					<< " " << b_dst.ref() << "\n";
#endif
				_pc.size_incr();
				return {iterator{*this, b_dst.sb()}, true};
			}
			catch ( const no_near_empty_bucket &e )
			{
				owner_lk.unlock();
#if TRACE_MANY
				std::cerr << "1. before resize\n" << make_table_dump(*this) << "\n";
#endif
				if ( segment_count() < _segment_capacity )
				{
					resize();
#if TRACE_MANY
					std::cerr << "2. after resize\n" << make_table_dump(*this) << "\n";
#endif
					goto RETRY;
				}
				throw;
			}
		}
		catch ( const perishable_expiry & )
		{
#if TRACE_PERISHABLE_EXPIRY
			std::cerr << "perishable expiry dump (emplace)\n"
				<< make_table_dump(*this)
				<< "\n";
#endif
			throw;
		}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::insert(
		const value_type &v_
	) -> std::pair<iterator, bool>
	try
	{
#if TRACE_MANY
		std::cerr << __func__ << " BEGIN LIST\n"
			<< make_table_dump(*this) << __func__
			<< " END LIST\n";
#endif
	RETRY:
		/* The bucket in which to place the new entry */
		auto sbw = make_segment_and_bucket(bucket(v_.first));
		auto owner_lk = make_owner_unique_lock(sbw);

		/* If the key already exists, refuse to insert */
		if ( auto cv = owner_lk.ref().value(owner_lk) )
		{
			auto sbc = sbw;
			for ( ; cv ; cv >>= 1U, sbc.incr(*this) )
			{
				if ( (cv & 1U) && key_equal()(locate(sbc).key(), v_.first) )
				{
#if TRACE_MANY
					std::cerr << __func__ << " (already present)\n";
#endif
					return {iterator{*this, sbc}, false};
				}
			}
		}

		/* the nearest free bucket */
		try
		{
			auto b_dst = nearest_free_bucket(sbw);
			b_dst = make_space_for_insert(owner_lk.index(), std::move(b_dst));

			b_dst.assert_clear(true, *this);
			b_dst.ref().content_move(v_, owner_lk.index());

			/* 3-step change to owner:
			 *  1. mark the content "ENTERING"
			 *   flush
			 *  2. atomically update the owner
			 *   flush
			 *  3. mark the content "IN_USE"
			 *   flush
			 */
			b_dst.ref().state_set(bucket_t::ENTERING);
			_pc.persist_content(b_dst.ref(), "content entering");
			_pc.size_destabilize();
			owner_lk.ref().insert(
				owner_lk.index()
				, unsigned(
					(
						b_dst.index() < owner_lk.index()
						? b_dst.index()+bucket_count()
						: b_dst.index()
					) - owner_lk.index()
				)
				, owner_lk
			);
			_pc.persist_owner(owner_lk.ref(), "owner insert");
			b_dst.ref().state_set(bucket_t::IN_USE);
			_pc.persist_content(b_dst.ref(), "content in_use");
#if TRACE_MANY
			std::cerr << __func__ << " bucket " << owner_lk.index()
				<< " store at " << b_dst.index() << " "
				<< make_owner_print(*this, owner_lk)
				<< " " << b_dst.ref() << "\n";
#endif
			_pc.size_incr();
			return {iterator{*this, b_dst.sb()}, true};
		}
		catch ( const no_near_empty_bucket &e )
		{
			owner_lk.unlock();
#if TRACE_MANY
			std::cerr << "1. before resize\n" << make_table_dump(*this) << "\n";
#endif

			if ( segment_count() < _segment_capacity )
			{
				resize();
#if TRACE_MANY
				std::cerr << "2. after resize\n" << make_table_dump(*this) << "\n";
#endif
				goto RETRY;
			}
			throw;
		}
	}
	catch ( const perishable_expiry & )
	{
#if TRACE_PERISHABLE_EXPIRY
		std::cerr << "perishable expiry dump (insert)\n"
			<< make_table_dump(*this) << "\n";
#endif
		throw;
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	void impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::resize()
	{
		_bc[segment_count()]._b = _pc.resize_prolog();
		_bc[segment_count()]._bucket_mutexes = new bucket_mutexes_t[bucket_count()];

		/* adjust count and everything which depends on it (size, mask) */

		/* PASS 1: copy content */
		resize_pass1();
		/* PASS 2: remove old content, update owners. Some old content mave have been
		 * removed if pass 2 was restarted, so use the new (junior) content to drive
		 * the operations.
		 */
		resize_pass2();

		_pc.resize_epilog();
	}

template <
	typename Key
	, typename T
	, typename Hash
	, typename Pred
	, typename Alloc
	, typename Mutex
>
	void impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::resize_pass1()
	{
		/* PASS 1: copy content */

		bix_t ix_senior = 0U;
		const auto sb_senior_end = make_segment_and_bucket_for_iterator(bucket_count());
		for (
			auto sb_senior = make_segment_and_bucket(0U)
			; sb_senior != sb_senior_end
			; sb_senior.incr_for_iterator(), ++ix_senior
		)
		{
			auto senior_content_lk = make_content_unique_lock(sb_senior);

			/* special locate, used to pre-fill new buckets */
			content<value_type> &junior_content = _bc[segment_count()]._b[ix_senior];
			if ( ! is_free(senior_content_lk.sb()) )
			{
				/* examine hash(key) to determine whether to copy content */
				auto hash = _hasher.hf(senior_content_lk.ref().key());
				auto ix_owner = bucket_expanded_ix(hash);
				/* [ix_owner, ix_owner + owner::size) is permissible range for content */
				if ( ix_owner <= ix_senior && ix_senior < ix_owner + owner::size )
				{
					/* content can stay where it is because bucket index index MSB is 0 */
#if TRACE_MANY
					std::cerr << __func__
						<< ".1a no-relocate owner " << bucket_ix(hash)
						<< " -> " << ix_owner << ": content " << ix_senior << "\n";
#endif
				}
				else if (
					ix_senior < owner::size
					&&
					bucket_count()*2U < ix_owner + owner::size
				)
				{
					/* content can stay where it is because the owner wraps
					 * NOTE: this test is not exact, but is close enough if owner::size
					 * is equal to or less than half the minimum table size.
					 */
#if TRACE_MANY
					std::cerr << __func__
						<< ".1b no-relocate owner " << bucket_ix(hash)
						<< " -> " << ix_owner << ": content " << ix_senior << "\n";
#endif
				}
				else
				{
					/* content must move */
					junior_content.content_share(senior_content_lk.ref(), ix_owner);
					senior_content_lk.ref().state_set(bucket_t::EXITING);
					junior_content.state_set(bucket_t::ENTERING);
#if TRACE_MANY
					std::cerr << __func__
						<< ".1c relocate owner " << bucket_ix(hash) << " -> " << ix_owner
						<< ": content " << ix_senior << " -> "
						<< ix_senior + bucket_count() << "\n";
#endif
				}
			}
		}

		/* flush state_EXITING for old content */
		_pc.persist_existing_segments("pass 1 owner+content seniors");
		/* persist new content */
		_pc.persist_new_segment("pass 1 copied content");
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	void impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::resize_pass2()
	{
		/* PASS 2: remove old content, update owners. Some old content mave have been
		 * removed if pass 2 was restarted, so use the new (junior) content to drive
		 * the operations.
		 */
		bix_t ix_senior = 0U;
		const auto sb_senior_end = make_segment_and_bucket_for_iterator(bucket_count());
		for (
			auto sb_senior = make_segment_and_bucket(0U)
			; sb_senior != sb_senior_end
			; sb_senior.incr_for_iterator(), ++ix_senior
		)
		{
			/* special locate, used before size has been updated to pre-fill new buckets */
			content_unique_lock_t
				junior_content_lk(
					_bc[segment_count()]._b[ix_senior]
					, segment_and_bucket(segment_count(), ix_senior)
					, _bc[segment_count()]._bucket_mutexes[ix_senior]._m_content
				);

			auto senior_content_lk = make_content_unique_lock(sb_senior);
			if ( junior_content_lk.ref().state_get() != bucket_t::FREE )
			{
				/* examine the key to locate old and new owners (owners) */
				auto hash = _hasher.hf(junior_content_lk.ref().key());
				auto ix_senior_owner = bucket_ix(hash);
				auto ix_junior_owner = bucket_expanded_ix(hash);
#if TRACE_MANY
				std::cerr << __func__ << ".2 content "
					<< ix_senior << " -> " << junior_content_lk.index()
					<< " owner " << ix_senior_owner << " -> " << ix_junior_owner
					<< "\n";
#endif
				if ( ix_senior_owner != ix_junior_owner )
				{
					auto senior_owner_sb = make_segment_and_bucket(ix_senior_owner);
					auto senior_owner_lk = make_owner_unique_lock(senior_owner_sb);
					owner_unique_lock_t
						junior_owner_lk(
							_bc[segment_count()]._b[ix_senior_owner]
							, segment_and_bucket(segment_count(), ix_senior_owner)
							, _bc[segment_count()]._bucket_mutexes[ix_senior_owner]._m_owner
						);

					/* special locate, used before size has been updated
					 * to pre-fill new buckets
					 */
					auto &junior_owner = _bc[segment_count()]._b[ix_senior_owner];
					unsigned owner_pos =
						static_cast<unsigned>(
							(
								ix_senior < ix_senior_owner
								? ix_senior + bucket_count()
								: ix_senior
							)
							-
							ix_senior_owner
						);
					assert(0 <= owner_pos);
					assert(owner_pos < owner::size);
					junior_owner.insert(ix_junior_owner, owner_pos, junior_owner_lk);
					senior_owner_lk.ref().erase(owner_pos, senior_owner_lk);
					_pc.persist_owner(junior_owner_lk.ref(), "pass 2 junior owner");
					_pc.persist_owner(senior_owner_lk.ref(), "pass 2 senior owner");
				}
				junior_content_lk.ref().state_set(bucket_t::IN_USE);
				_pc.persist_content(junior_content_lk.ref());
				senior_content_lk.ref().erase();
				senior_content_lk.ref().state_set(bucket_t::FREE);
			}
			else if ( senior_content_lk.ref().state_get() != bucket_t::FREE )
			{
#if TRACK_OWNER
				bool owner_update_owed = false;
				if ( ix_senior < senior_content_lk.ref().owner() )
				{
					/* Wrap. Should only occur where owner is near end of table and
					 * content is near beginning
					 */
					assert(ix_senior < base_segment_size);
					/* adjust the owner location (as kept in content) to reflect the
					 * new owner location
					 */
					owner_update_owed = true;
				}
#endif
				/* examine the key to locate old and new owners (owners) */
				auto hash = _hasher.hf(senior_content_lk.ref().key());
				auto ix_senior_owner = bucket_ix(hash);
				auto ix_junior_owner = bucket_expanded_ix(hash);
				if ( ix_senior_owner != ix_junior_owner )
				{
					/* content not moved, but owner changes (because content
					 * is near start of table and owner moves from near old end
					 * to near new end)
					 */
					auto senior_owner_sb = make_segment_and_bucket(ix_senior_owner);
					auto senior_owner_lk = make_owner_unique_lock(senior_owner_sb);
					owner_unique_lock_t
						junior_owner_lk(
							_bc[segment_count()]._b[ix_senior_owner]
							, segment_and_bucket(segment_count(), ix_senior_owner)
							, _bc[segment_count()]._bucket_mutexes[ix_senior_owner]._m_owner
						);

					/* special locate, used before size has been updated
					 * to pre-fill new buckets
					 */
					auto &junior_owner = _bc[segment_count()]._b[ix_senior_owner];
					unsigned owner_pos =
						static_cast<unsigned>(
							(
								ix_senior < ix_senior_owner
								? ix_senior + bucket_count()
								: ix_senior
							)
							-
							ix_senior_owner
						);
					assert(0 <= owner_pos);
					assert(owner_pos < owner::size);
					junior_owner.insert(ix_junior_owner, owner_pos, junior_owner_lk);
					senior_owner_lk.ref().erase(owner_pos, senior_owner_lk);
					assert(ix_senior < base_segment_size);
#if TRACK_OWNER
					/* adjust the owner location (as kept in content) to reflect the
					 * new owner location
					 */
					senior_content_lk.ref().owner_update(bucket_count());
					owner_update_owed = false;
#endif
					_pc.persist_owner(junior_owner_lk.ref(), "pass 2 junior owner");
					_pc.persist_owner(senior_owner_lk.ref(), "pass 2 senior owner");
				}
#if TRACK_OWNER
				assert(!owner_update_owed);
#endif
			}
		}
		for (
			auto ix_senior_owner = 0U
			; ix_senior_owner != bucket_count()
			; ++ix_senior_owner
		)
		{
			/* special locate, used before size has been updated
			 * to pre-fill new buckets
			 */
			owner_unique_lock_t
				junior_owner_lk(
					_bc[segment_count()]._b[ix_senior_owner]
					, segment_and_bucket(segment_count(), ix_senior_owner)
					, _bc[segment_count()]._bucket_mutexes[ix_senior_owner]._m_owner
				);
		}

		/* flush for state_set bucket_t::FREE in loop above. */
		_pc.persist_existing_segments("pass 2 senior content");
		/* flush for state_set owner::LIVE in loop above. */
		_pc.persist_new_segment("pass 2 junior owner");
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::make_segment_and_bucket(
		bix_t ix_
	) const -> segment_and_bucket
	{
		assert( ix_ < bucket_count() );
		return make_segment_and_bucket_for_iterator(ix_);
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::make_segment_and_bucket_prev(
		segment_and_bucket a
		, unsigned bkwd
	) const -> segment_and_bucket
	{
		a.subtract_small(*this, bkwd);
		return a;
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::make_segment_and_bucket_for_iterator(
		bix_t ix
	) const -> segment_and_bucket
	{
		assert( ix <= bucket_count() );
		return segment_and_bucket(ix);
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::locate(
		const segment_and_bucket &a_
	) const -> bucket_aligned_t &
	{
		return const_cast<bucket_aligned_t &>(_bc[a_.si()]._b[a_.bi()]);
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::locate_bucket_mutexes(
		const segment_and_bucket &a_
	) const -> bucket_mutexes_t &
	{
		return _bc[a_.si()]._bucket_mutexes[a_.bi()];
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::locate_owner(
		const segment_and_bucket &a_
	) const -> const owner &
	{
		return static_cast<const owner &>(locate(a_));
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::locate_content(
		const segment_and_bucket &a_
	) const -> const content_t &
	{
		return static_cast<const content_t &>(locate(a_));
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::locate_content(
		const segment_and_bucket &a_
	) -> content_t &
	{
		return static_cast<content_t &>(locate(a_));
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::make_owner_unique_lock(
		const segment_and_bucket &a_
	) const -> owner_unique_lock_t
	{
		return owner_unique_lock_t(locate(a_), a_, locate_bucket_mutexes(a_)._m_owner);
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::make_owner_unique_lock(
		const content_unique_lock_t &cl_
		, unsigned bkwd
	) const -> owner_unique_lock_t
	{
		auto a = cl_.sb();
		a.subtract_small(*this, bkwd);
		return owner_unique_lock_t(locate(a), a, locate_bucket_mutexes(a)._m_owner);
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::make_owner_shared_lock(
		const key_type &k_
	) const -> owner_shared_lock_t
	{
		auto a = make_segment_and_bucket(bucket(k_));
		return owner_shared_lock_t(locate(a), a, locate_bucket_mutexes(a)._m_owner);
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::make_content_unique_lock(
		const segment_and_bucket &a_
	) const -> content_unique_lock_t
	{
		return content_unique_lock_t(locate(a_), a_, locate_bucket_mutexes(a_)._m_content);
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::make_content_unique_lock(
		const owner_unique_lock_t &wl_
		, unsigned fwd
	) const -> content_unique_lock_t
	{
		auto a = wl_.sb();
		a.add_small(*this, fwd);
		return content_unique_lock_t(locate(a), a, locate_bucket_mutexes(a)._m_content);
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	template <typename Lock>
		auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::locate_key(
			Lock &bi_
			, const key_type &k_
		) const -> std::tuple<bucket_t *, segment_and_bucket>
		{
			/* Use the owner to filter key checks, a performance aid
			 * to reduce the number of key compares.
			 */
			auto wv = bi_.ref().value(bi_);
#if TRACE_MANY
			std::cerr
				<< __func__
				<< " owner "
				<< make_owner_print(*this, bi_)
				<< " value " << wv
				<< "\n";
#endif
			auto bfp = bi_.sb();
			auto &t =
				*const_cast<table_base<Key, T, Hash, Pred, Alloc, Mutex> *>(this);
			++t._locate_key_call;
			for ( auto content_offset = 0U; content_offset != owner::size; ++content_offset )
			{
				if ( ( wv & 1 ) == 1 )
				{
					++t._locate_key_owned;
					auto c = &locate(bfp);
					if ( key_equal()(c->key(), k_) )
					{
						++t._locate_key_match;
#if TRACE_MANY
						std::cerr
							<< __func__ << " returns (success) " << bfp.index() << "\n";
#endif
						bucket_t *bb = static_cast<bucket_t *>(c);
						return std::tuple<bucket_t *, segment_and_bucket>(bb, bfp);
					}
					else
					{
						++t._locate_key_mismatch;
					}
				}
				{
					++t._locate_key_unowned;
				}
				bfp.incr(*this);
				wv >>= 1U;
			}
#if TRACE_MANY
			std::cerr
				<< __func__ << " returns (failure) " << bfp.index() << "\n";
#endif
			return
				std::tuple<bucket_t *, segment_and_bucket>(
					nullptr
					, segment_and_bucket(0, 0)
				);
		}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::erase(
		const key_type &k_
	) -> size_type
	try
	{
		/* The bucket which owns the entry */
		auto sbw = make_segment_and_bucket(bucket(k_));
		auto owner_lk = make_owner_unique_lock(sbw);
		const auto erase_ix = locate_key(owner_lk, k_);
		if ( std::get<0>(erase_ix) == nullptr )
		{
			/* no such element */
			return 0U;
		}
		else /* element found at bf */
		{
			auto erase_src = make_content_unique_lock(std::get<1>(erase_ix));
			/* 3-step owner erase
			 * 1. mark content EXITING
			 *  flush
			 * 2. disclaim owner ownership atomically
			 *  flush
			 * 3. mark content FREE
			 *  flush
			 */
			erase_src.ref().state_set(bucket_t::EXITING);
			_pc.persist_content(erase_src.ref(), "content erase exiting");
			_pc.size_destabilize();
			owner_lk.ref().erase(
				static_cast<unsigned>(erase_src.index()-owner_lk.index())
				, owner_lk
			);
			_pc.persist_owner(owner_lk.ref(), "owner erase");
			erase_src.ref().erase();
			_pc.persist_content(erase_src.ref(), "content erase free");
			_pc.size_decr();
			return 1U;
		}
	}
	catch ( const perishable_expiry & )
	{
#if TRACE_PERISHABLE_EXPIRY
		std::cerr << "perishable expiry dump (erase)\n"
			<< make_table_dump(*this) << "\n";
#endif
		throw;
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::count(
		const key_type &k_
	) const -> size_type
	{
		auto bi_lk = make_owner_shared_lock(k_);
		const auto bf = locate_key(bi_lk, k_);
#if TRACE_MANY
		std::cerr << __func__
			<< " " << k_
			<< " starting at " << bi_lk.index()
			<< " "
			<< make_owner_print(*this, bi_lk)
			<< " found "
			<< *bf << "\n";
#endif
		return bf ? 1U : 0U;
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::at(
		const key_type &k_
	) const -> const mapped_type &
	{
		/* The bucket which owns the entry */
		auto bi_lk = make_owner_shared_lock(k_);
		const auto bf = std::get<0>(locate_key(bi_lk, k_));
		if ( ! bf )
		{
			/* no such element */
			throw std::out_of_range("no such element");
		}
		/* element found at bf */
		return bf->mapped();
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::at(
		const key_type &k_
	) -> mapped_type &
	{
		/* Lock the entry owner */
		auto bi_lk = make_owner_shared_lock(k_);
		const auto bf = std::get<0>(locate_key(bi_lk, k_));
		if ( ! bf )
		{
			/* no such element */
			throw std::out_of_range("no such element");
		}
		/* element found at bf */
		return bf->mapped();
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::bucket(
		const key_type &k_
	) const -> size_type
	{
		return bucket_ix(_hasher.hf(k_));
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::bucket_size(
		size_type n_
	) const -> size_type
	{
		auto a = make_segment_and_bucket(n_);
		auto g = owner_shared_lock_t(locate(a), a, locate_bucket_mutexes(a)._m_owner);
		size_type s = 0;
		for ( auto v = g.ref().get_value(g); v; v >>= 1U )
		{
			s += ( v && 1u );
		}
		return s;
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::lock_shared(
		const key_type &k_
	) -> bool
	{
		/* Lock the entry owner */
		auto bi_lk = make_owner_shared_lock(k_);
		const auto bf = locate_key(bi_lk, k_);

		if ( std::get<0>(bf) == nullptr )
		{
			/* no such element */
			return false;
		}

		auto &m = locate_bucket_mutexes(std::get<1>(bf));
		auto b = m._m_content.try_lock_shared();
		if ( b )
		{
			m._state = bucket_mutexes_t::SHARED;
		}
		return b;
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::lock_unique(
		const key_type &k_
	) -> bool
	{
		/* Lock the entry owner */
		auto bi_lk = make_owner_shared_lock(k_);
		const auto bf = locate_key(bi_lk, k_);

		if ( std::get<0>(bf) == nullptr )
		{
			/* no such element */
			return false;
		}

		auto &m = locate_bucket_mutexes(std::get<1>(bf));
		auto b = m._m_content.try_lock();
		if ( b )
		{
			m._state = bucket_mutexes_t::UNIQUE;
		}
		return b;
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::unlock(
		const key_type &k_
	) -> void
	{
		/* Lock the entry owner */
		auto bi_lk = make_owner_shared_lock(k_);
		const auto bf = locate_key(bi_lk, k_);

		if ( std::get<0>(bf) != nullptr )
		{
			/* found an element */
			auto &m = locate_bucket_mutexes(std::get<1>(bf));
			if ( m._state == bucket_mutexes_t::UNIQUE )
			{
				m._state = bucket_mutexes_t::SHARED;
				m._m_content.unlock();
			}
			else
			{
				m._m_content.unlock_shared();
			}
		}
	}

template <typename Key, typename T, typename Hash, typename Pred, typename Alloc, typename Mutex>
	auto impl::table_base<Key, T, Hash, Pred, Alloc, Mutex>::size(
	) const -> size_type
	{
		return _pc.size();
	}
