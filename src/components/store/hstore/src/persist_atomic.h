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


#ifndef _COMANCHE_HSTORE_PERSIST_ATOMIC_H
#define _COMANCHE_HSTORE_PERSIST_ATOMIC_H

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
			using mod_key_t = persist_fixed_string<char, Value::first_type::small_size, typename Value::first_type::allocator_type>;
			mod_key_t mod_key;
			/* source of modification data */
			persist_fixed_string<char, Value::second_type::small_size, typename Value::second_type::allocator_type> mod_mapped;
			/* control of modification data */
			persistent_t<mod_ctl_ptr_t> mod_ctl;
			/* size of control located by mod_ctl (0 if no outstanding modification, negative if the modfication is a replace by erase/emplace */
			persistent_atomic_t<std::ptrdiff_t> mod_size;
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
