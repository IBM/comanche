/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef _COMANCHE_HSTORE_ATOMIC_CTL_H_
#define _COMANCHE_HSTORE_ATOMIC_CTL_H_

#include "construction_mode.h"
#include "persist_atomic.h"
#include "persist_fixed_string.h"

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

			using persist_t = persist_atomic<typename Table::value_type>;
			using mod_key_t = typename persist_t::mod_key_t;
			persist_t *_persist; /* persist_atomic is a bad name. Should be a noun. */
			table_t *_map;
			void redo_update();
			void redo_replace();
			void redo_finish();
		public:
			atomic_controller(
				persist_atomic<typename Table::value_type> &persist_
				, table_t &map_
				, construction_mode mode_
			);
			atomic_controller(const atomic_controller &) = delete;
			atomic_controller& operator=(const atomic_controller &) = delete;

			void redo();

			void persist_range(const void *first_, const void *last_, const char *what_);

			auto enter_update(
				typename Table::allocator_type al_
				, typename Table::key_type &key
				, std::vector<Component::IKVStore::Operation *>::const_iterator first
				, std::vector<Component::IKVStore::Operation *>::const_iterator last
			) -> Component::status_t;
			auto enter_replace(
				typename Table::allocator_type al_
				, typename Table::key_type &key
				, const char *data_
				, std::size_t data_len_
			) -> Component::status_t;
	};
}

#include "atomic_controller.tcc"

#endif
