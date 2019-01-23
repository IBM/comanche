#ifndef _DAWN_PERSIST_MAP_H
#define _DAWN_PERSIST_MAP_H

#include "bucket_aligned.h"
#include "hash_bucket.h"
#include "persist_fixed_string.h"
#include "persistent.h"
#include "persist_atomic.h"
#include "segment_layout.h"

#include <cassert>
#include <cstddef> /* size_t */
#include <limits> /* numeric_limits */

/* Persistent data for hstore.
 */

namespace impl
{
	template <typename Allocator>
		class persist_controller;

	class size_control
	{
		/* unstable == 0 implies that size is valid and persisted
		 * if unstable != 0, restart must individually count the contents.
		 *
		 * The size_and_unstable field is 2^N times the actual size,
		 * and the lower N bits represents "unstable," not part of the size.
		 */
		persistent_atomic_t<std::size_t> _size_and_unstable;
		static constexpr unsigned N = 8;
		static constexpr std::size_t count_1 = 1U<<N;
		std::size_t destable_count() const { return _size_and_unstable & (count_1-1U); }
	public:
		size_control()
			: _size_and_unstable(0)
		{}
		void size_set_stable(std::size_t n) { _size_and_unstable = (n << N); }
		std::size_t size() const { assert( is_stable() ); return _size_and_unstable >> N; }
		void stabilize() { assert(destable_count() != 0); --_size_and_unstable; }
		void destabilize() { ++_size_and_unstable; assert(destable_count() != 0); }
		void decr() { _size_and_unstable -= count_1; }
		void incr() { _size_and_unstable += count_1; }
		bool is_stable() const { return destable_count() == 0; }
	};

	/* Size change state as an object. Not for RIAA strictness, but to move a memory
	 * fetch which appears to stall the pipeline following a call to pmem_persist.
	 */
	class size_change
	{
		size_control *_size_control;
		size_change(const size_change &) = delete;
		size_change& operator=(const size_change &) = delete;
	protected:
		size_control & get_size_control() const { return *_size_control; }
	public:
		size_change(size_control &ctl)
			: _size_control(&ctl)
		{
			_size_control->destabilize();
		}
		~size_change()
		{
			_size_control->stabilize();
		}
	};

	class size_incr
		: public size_change
	{
	public:
		size_incr(size_control &ctl)
			: size_change(ctl)
		{}
		~size_incr()
		{
			get_size_control().incr();
		}
	};

	class size_decr
		: public size_change
	{
	public:
		size_decr(size_control &ctl)
			: size_change(ctl)
		{}
		~size_decr()
		{
			get_size_control().decr();
		}
	};

	template <typename Allocator>
		class persist_map
		{
			using value_type = typename Allocator::value_type;
			using bucket_aligned_t = bucket_aligned<hash_bucket<value_type>>;
			using bucket_allocator_t =
				typename Allocator::template rebind<bucket_aligned_t>::other;
			using bucket_ptr = typename bucket_allocator_t::pointer;
			/* bucket indexes */
			using bix_t = segment_layout::bix_t;
			/* segment indexes */
			using six_t = segment_layout::six_t;

			struct segment_count
			{
				/* current segment count */
				persistent_atomic_t<six_t> _actual;
				/* desired segment count */
				persistent_atomic_t<six_t> _target;
				segment_count(six_t target_)
					: _actual(0)
					, _target(target_)
				{}
			};

			struct segment_control
			{
				persistent_t<bucket_ptr> bp;
				segment_control()
					: bp()
				{
				}
			};

			static constexpr six_t _segment_capacity = 32U;
			static constexpr unsigned log2_base_segment_size =
				segment_layout::log2_base_segment_size;
			static constexpr bix_t base_segment_size =
				segment_layout::base_segment_size;

			size_control _size_control;

			segment_count _segment_count;

			segment_control _sc[_segment_capacity];

		public:
			persist_map(std::size_t n, Allocator av);
			friend class persist_controller<Allocator>;
		};
}

#include "persist_map.tcc"

#endif
