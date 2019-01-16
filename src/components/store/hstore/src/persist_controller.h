#ifndef _DAWN_PERSIST_CONTROLLER_H
#define _DAWN_PERSIST_CONTROLLER_H

#include "persist_data.h"
#include "persist_switch.h"

#include <boost/iterator/transform_iterator.hpp>

#include <cassert>
#include <cstddef> /* size_t */
#include <functional> /* mem_fn */
#include <type_traits>

/* Controller for persistent data.
 * Modification of persistent data (except for writes to persist_data::_sc)
 * goes through this class. Ideally this should also get writes to persist_data::_sc.
 */

namespace impl
{
	template <typename Allocator>
		class persist_controller
			: public Allocator
		{
			using persist_data_t = persist_map<Allocator>;
			using value_type = typename Allocator::value_type;
		public:
			using size_type = std::size_t;
			using bix_t = std::size_t; /* sufficient for all bucket indexes */
			using bucket_aligned_t = bucket_aligned<hash_bucket<value_type>>;
			using content_t = content<value_type>;
			static constexpr auto _segment_capacity = persist_data_t::_segment_capacity;
			static constexpr unsigned log2_base_segment_size =
				persist_data_t::log2_base_segment_size;
			static constexpr bix_t base_segment_size = persist_data_t::base_segment_size;
		private:
			using bucket_allocator_t = typename persist_data_t::bucket_allocator_t;
			persist_data_t *_persist;
			std::size_t _bucket_count_cached;
			/* enable persist call if Allocator supports persist */
			using persist_switch_t =
				persist_switch<Allocator, std::is_base_of<persister, Allocator>::value>;

			void persist_segment_table(); /* Flush the bucket pointers (*_b) */
			void persist_internal(
				const void *first
				, const void *last
				, const char *what
			);
			void size_stabilize();
			auto bucket_count_uncached() -> size_type
			{
				return base_segment_size << (segment_count_actual() - 1U);
			}

		public:
			explicit persist_controller(const Allocator &av, persist_data_t *persist);

			persist_controller(const persist_controller &) = delete;
			auto operator=(
				const persist_controller &
			) -> persist_controller & = delete;

			auto resize_prolog() -> bucket_aligned_t *;
			void resize_epilog();

			void size_destabilize();
			void size_incr();
			void size_decr();

			void persist_owner(
				const owner &b
				, const char *what = "bucket_owner"
			);
			void persist_content(
				const content_t &b
				, const char *what = "bucket_content"
			);
			void persist_segment_count(); /* Flush the bucket pointer count (_count) */
			void persist_size();
			void persist_existing_segments(const char *what = "old segments");
			void persist_new_segment(const char *what = "new segments");

			std::size_t segment_count_actual() const
			{
				return _persist->_segment_count._actual;
			}
			std::size_t segment_count_target() const
			{
				return _persist->_segment_count._target;
			}
			std::size_t size() const
			{
				return _persist->_size_control.size();
			}

			/* NOTE: this function returns an non-const iterator over _persist data,
			 * allowing successive accesses to *_persist without intervening "ticks."
			 * The user needs to provide intervening ticks.
			 */
			auto bp_src()
			{
				return boost::make_transform_iterator(
					_persist->_sc
					, std::mem_fn(&persist_data_t::segment_control::bp)
				);
			}
			bool is_size_unstable() const;
			void size_set(std::size_t n);

			auto bucket_count() const -> size_type
			{
				return _bucket_count_cached;
			}

			auto max_bucket_count() const -> size_type
			{
				return base_segment_size << (_segment_capacity - 1U);
			}

			auto mask() const { return bucket_count() - 1U; }

			auto distance_wrapped(
				bix_t first, bix_t last
			) -> unsigned
			{
				return
					unsigned
					(
						(
							last < first
							? last + bucket_count()
							: last
						) - first
					);
			}
		};
}

#endif
