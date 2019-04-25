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


#ifndef _COMANCHE_HSTORE_CONTENT_H
#define _COMANCHE_HSTORE_CONTENT_H

#include "trace_flags.h"

#include "hop_hash_log.h"
#include "perishable.h"
#include "persistent.h"
#if TRACED_CONTENT
#include "hop_hash_debug.h"
#endif

#include <cassert>
#include <cstddef> /* size_t */
#include <limits> /* numeric_limits */
#include <string>

/*
 * content
 */

namespace impl
{
	template <typename Value>
		class content
		{
		public:
			enum state_t { FREE, IN_USE };
		private:
			using key_t = typename Value::first_type;
			using mapped_t = typename Value::second_type;
			using value_t = Value;
			persistent_atomic_t<state_t> _state;
			/* NOTE: Cannot make _value persistent, but the user can make value's
			 * individual conponents persistent.
			 */
			value_t _value;
			using owner_t = std::size_t; /* sufficient for all bucket indexes */
			void set_owner(owner_t);
			owner_t get_owner() const
			{
				return
#if TRACK_OWNER
					_owner
#else
					owner_t()
#endif
					;
			}
			static constexpr auto owner_undefined = std::numeric_limits<owner_t>::max();
#if TRACK_OWNER
			owner_t _owner; /* remember the bucket which owns this content */
#endif
#if TRACED_CONTENT
			auto descr() const -> std::string;
			auto to_string() const -> std::string;
			auto state_string() const -> std::string;
#endif
		public:
			explicit content();
			content(const content &) = delete;
			content &operator=(const content &) = delete;
			~content()
			{
				if ( _state != FREE )
				{
					_value.~value_t();
				}
			}

			template <typename ... Args>
				auto content_construct(
					std::size_t bi
					, Args && ... args
				) -> content &;

			auto content_share(content &from) -> content &;

			auto content_share(
				const content &sr
				, std::size_t bi
			) -> content &;

			const key_t &key() const { return _value.first; }
			const mapped_t &mapped() const { return _value.second; }
			/* PMEM ESCAPE: Uncontrolled access to _value: only used by at(), which is not
			 * itself used internally
			 */
			mapped_t &mapped() { return _value.second; }
			/* PMEM ESCAPE: Uncontrolled access to _value: only used by iterator, which is
			 * not itself used internally
			 */
			value_t &value() { return _value; }
		public:
			auto erase() -> void;
			void state_set(state_t state_)
			{
				_state = state_;
			}
			auto state_get() const -> state_t { return _state; }
			auto is_clear() const noexcept -> bool;
			template <typename Lock, typename Table>
				void assert_clear(bool b, Lock &lk, Table &t)
				{
#if TRACED_CONTENT && 0
					/* NOTE: Disabled. Not reliable if a crash has occurred. A bucket
					 * may incorrectly appear to have an owner, and therefore content,
					 * if a crash occurred while it it was being populated.
					 */
					if ( ! is_clear() )
					{
						hop_hash_log::write(__func__
							, " assert_clear fail: bucket ", lk.index()
							, " has content "
							, lk.ref()
							, "\n"
							, dump<TRACED_CONTENT && 0>::make_table_dump(t)
						);
					}
					assert(is_clear() == b);
#else
					(void)(b);
					(void)(lk);
					(void)(t);
#endif
				}

#if TRACK_OWNER
			void owner_verify(owner_t owner) const;
			owner_t owner() const { return _owner; }
			void owner_update(owner_t owner_delta);
#endif
#if TRACED_CONTENT
			friend auto operator<< <>(
				std::ostream &o
				, const content<Value> &c
			) -> std::ostream &;
#endif
		};
}

#include "content.tcc"

#endif
