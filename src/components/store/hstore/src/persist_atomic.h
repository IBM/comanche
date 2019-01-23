#ifndef _DAWN_PERSIST_ATOMIC_H
#define _DAWN_PERSIST_ATOMIC_H

#include "persist_fixed_string.h"
#include "persistent.h"

#include <cstddef> /* size_t */

/* Persistent data for hstore.
 */

namespace impl
{
#if 0
	/* Until we get friendship sorted out.
	 * The atomic_controller needs a class specialized by allocator only
	 * to be friends with persist_atomic
	 */
	template <typename Allocator>
		class atomic_controller;
#endif

	struct mod_control
	{
		persistent_t<std::size_t> offset_src;
		persistent_t<std::size_t> offset_dst;
		persistent_t<std::size_t> size;
		explicit mod_control(std::size_t s, std::size_t d, std::size_t z)
			: offset_src(s)
			, offset_dst(d)
			, size(z)
		{}
		explicit mod_control() : mod_control(0, 0, 0) {}
	};

	template <typename Value>
		class persist_atomic
		{
#if 0
#else
		public:
#endif
			using allocator_type = typename Value::first_type::allocator_type;
			using mod_ctl_ptr_t = typename allocator_type::template rebind<mod_control>::other::pointer;

			/* key to destination of modification data */
			persist_fixed_string<char, typename Value::first_type::allocator_type> mod_key;
			/* source of modification data */
			persist_fixed_string<char, typename Value::second_type::allocator_type> mod_mapped;
			/* control of modification datai */
			persistent_t<mod_ctl_ptr_t> mod_ctl;
			/* size of control located by mod_ctl (0 if no outstanding modification) */
			persistent_atomic_t<std::size_t> mod_size;
		public:
			persist_atomic()
				: mod_key()
				, mod_mapped()
				, mod_ctl()
				, mod_size(0U)
			{
			}
			persist_atomic(const persist_atomic &) = delete;
			persist_atomic& operator=(const persist_atomic &) = delete;
#if 0
			friend class atomic_controller<Allocator>;
#endif
		};
}

#endif
