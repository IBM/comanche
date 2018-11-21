#ifndef _DAWN_HSTORE_ATOMIC_CTL_H_
#define _DAWN_HSTORE_ATOMIC_CTL_H_

#include "persist_atomic.h"
#include "persist_fixed_string.h"
#include "persist_switch.h"
#include "persister.h"

#include <type_traits> /* is_base_of */
#include <vector>

namespace impl
{
	template <typename Table>
		class atomic_controller
			: private Table::allocator_type::template rebind<mod_control>::other
		{
			using table_t = Table;
			using allocator_type =
				typename table_t::allocator_type::template rebind<mod_control>::other;
			using persist_switch_t =
				persist_switch<allocator_type, std::is_base_of<persister, allocator_type>::value>;
			persist_atomic<allocator_type> *_persist; /* Bad name. Should be a noun. */
			table_t  *_map;
		public:
			atomic_controller(
				persist_atomic<allocator_type> &persist_
				, table_t &map_
			);
			atomic_controller(const atomic_controller &) = delete;
			atomic_controller& operator=(const atomic_controller &) = delete;

			void redo();

			void persist_range(const void *first_, const void *last_, const char *what_);

			auto enter(
				PMEMobjpool *pop
				, persist_fixed_string<char> &key
				, uint64_t type_num_data
				, std::vector<Component::IKVStore::operation_t>::const_iterator first
				, std::vector<Component::IKVStore::operation_t>::const_iterator last
			) -> Component::status_t;
	};
}

#include "atomic_controller.tcc"

#endif