#ifndef _COMANCHE_HSTORE_OWNER_H
#define _COMANCHE_HSTORE_OWNER_H

#include "trace_flags.h"

#include "persistent.h"
#if TRACE_OWNER
#include "hop_hash_debug.h"
#endif

#include <cassert>
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <limits> /* numeric_limits */
#include <string>

#if TRACE_OWNER
#include <iostream>
#endif

/*
 * The "owner" part of a hash bucket
 */

namespace impl
{
	template <typename Bucket, typename Referent, typename Lock>
		struct bucket_shared_lock;
	template <typename Bucket, typename Referent, typename Lock>
		using bucket_shared_ref = bucket_shared_lock<Bucket, Referent, Lock> &;

	template <typename Bucket, typename Referent, typename Lock>
		struct bucket_unique_lock;
	template <typename Bucket, typename Referent, typename Lock>
		using bucket_unique_ref = bucket_unique_lock<Bucket, Referent, Lock> &;

	class owner
	{
	public:
		static constexpr unsigned size = 32U;
		using value_type = std::uint64_t; /* sufficient for size not over 64U */
		static constexpr auto pos_undefined = std::numeric_limits<std::size_t>::max();
	private:
		persistent_atomic_t<value_type> _value; /* at least owner::size bits */
#if TRACK_POS
		std::size_t _pos;
#endif
	public:
		explicit owner()
			: _value(0)
#if TRACK_POS
			, _pos(pos_undefined)
#endif
		{}

		template<typename Bucket, typename Referent, typename SharedMutex>
			void insert(
				const std::size_t
#if TRACK_POS
					pos_
#endif
				, const unsigned p_
				, bucket_unique_ref<Bucket, Referent, SharedMutex>
			)
		{
#if TRACK_POS
			assert(_pos == pos_undefined || _pos == pos_);
			assert(p_ < size);
			_pos = pos_;
#endif
			_value |= (1U << p_);
		}
		template<typename Bucket, typename Referent, typename SharedMutex>
			void erase(unsigned p, bucket_unique_ref<Bucket, Referent, SharedMutex>)
			{
				_value &= ~(1U << p);
			}
		template<typename Bucket, typename Referent, typename SharedMutex>
			void move(
				unsigned dst_
				, unsigned src_
				, bucket_unique_ref<Bucket, Referent, SharedMutex>
			)
			{
				assert(dst_ < size);
				assert(src_ < size);
				_value = (_value | (1U << dst_)) & ~(1U << src_);
			}
		template <typename Lock>
			auto value(Lock &) const -> value_type { return _value; }
		template <typename Lock>
			auto owned(std::size_t table_size, Lock &) const -> std::string;
		/* clear the senior owner of all the bits set in its new junior owner. */
		template <typename Bucket, typename Referent, typename SharedMutex>
			void clear_from(
				const owner &junior
				, bucket_unique_ref<Bucket, Referent, SharedMutex>
				, bucket_shared_ref<Bucket, Referent, SharedMutex>
			)
			{
				_value &= ~junior._value;
			}

#if TRACE_OWNER
		template <
			typename Lock
		>
			friend auto operator<<(
				std::ostream &o
				, const impl::owner_print<Lock> &
			) -> std::ostream &;
		template <typename Table>
			friend auto operator<<(
				std::ostream &o
				, const impl::owner_print<impl::bypass_lock<const typename Table::bucket_t, const impl::owner>> &
			) -> std::ostream &;
#endif
	};
}

#endif
