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


#ifndef COMANCHE_HSTORE_SESSION_H
#define COMANCHE_HSTORE_SESSION_H

#include "hstore_tracked_pool.h"

#include "construction_mode.h"

#include <utility> /* move */
#include <vector>

/* open_pool_handle, ALLOC_T, table_t */
template <typename Handle, typename Allocator, typename Table>
	class session
		: public tracked_pool
	{
		Handle _pop;
		Allocator _heap;
		Table _map;
		impl::atomic_controller<Table> _atomic_state;

		class lock_impl
			: public Component::IKVStore::Opaque_key
		{
			std::string _s;
		public:
			lock_impl(const std::string &s_)
			: Component::IKVStore::Opaque_key{}
			, _s(s_)
			{}
			const std::string &key() const { return _s; }
		};

		static bool try_lock(table_t &map, hstore::lock_type_t type, const KEY_T &p_key)
		{
			return
				type == Component::IKVStore::STORE_LOCK_READ
				? map.lock_shared(p_key)
				: map.lock_unique(p_key)
				;
		}

		class definite_lock
		{
			table_t &_map;
			const KEY_T &_key;
		public:
			definite_lock(table_t &map_, const KEY_T &pkey_)
				: _map(map_)
				, _key(pkey_)
			{
				if ( ! _map.lock_unique(_key) )
				{
					throw General_exception("unable to get write lock");
				}
			}
			~definite_lock()
			{
				_map.unlock(_key); /* release lock */
			}
		};

		/* Return value not set. Ignored?? */
		static int _functor(
			const std::string &key
			, MAPPED_T &m
			, std::function
			<
				int(const std::string &key, const void *val, std::size_t val_len)
			> *lambda
		)
		{
			assert(lambda);
			(*lambda)(key, m.data(), m.size());
			return 0;
		}

		auto allocator() const { return _heap; }
		table_t &map() noexcept { return _map; }
		const table_t &map() const noexcept { return _map; }

		auto enter_update(
			typename Table::key_type &key
			, std::vector<Component::IKVStore::Operation *>::const_iterator first
			, std::vector<Component::IKVStore::Operation *>::const_iterator last
			) -> Component::status_t
		{
			return _atomic_state.enter_update(allocator(), key, first, last);
		}

		auto enter_replace(
			typename Table::key_type &key
			, const void *data
			, std::size_t data_len
		) -> Component::status_t
		{
			return _atomic_state.enter_replace(allocator(), key, static_cast<const char *>(data), data_len);
		}
	public:
		/* PMEMoid, persist_data_t */
		template <typename OID, typename Persist>
			explicit session(
				OID
#if USE_CC_HEAP == 1 || USE_CC_HEAP == 2
					heap_oid_
#endif
				, const pool_path &path_
				, open_pool_handle &&pop_
				, Persist *persist_data_
			)
			: tracked_pool(path_)
			, _pop(std::move(pop_))
			, _heap(
				Allocator(
#if USE_CC_HEAP == 0
					this->pool()
#elif USE_CC_HEAP == 1
					*new
						(pmemobj_direct(heap_oid_))
						heap_cc(static_cast<char *>(pmemobj_direct(heap_oid_) + sizeof(heap_cc)))
#elif USE_CC_HEAP == 2
					*new
						(pmemobj_direct(heap_oid_))
						heap_co(heap_oid_)
#else /* USE_CC_HEAP */
					this->pool() /* not used */
#endif /* USE_CC_HEAP */
					)
			)
			, _map(persist_data_, _heap)
			, _atomic_state(*persist_data_, _map)
		{}

		explicit session(
			const pool_path &path_
			, Handle &&pop_
			, construction_mode mode_
		)
			: tracked_pool(path_)
			, _pop(std::move(pop_))
			, _heap(
				Allocator(
					this->pool()->heap
				)
			)
			, _map(&this->pool()->persist_data, mode_, _heap)
			, _atomic_state(this->pool()->persist_data, _map, mode_)
		{}

		session(const session &) = delete;
		session& operator=(const session &) = delete;
		/* session constructor and get_pool_regions only */
		auto *pool() const { return _pop.get(); }
	public:

		auto insert(
			const std::string &key,
			const void * value,
			const std::size_t value_len
		)
		{
			auto cvalue = static_cast<const char *>(value);

			return
#if 1
				map().emplace(
					std::piecewise_construct
					, std::forward_as_tuple(key.begin(), key.end(), this->allocator())
					, std::forward_as_tuple(cvalue, cvalue + value_len, this->allocator())
				);
#else
				map().insert(
					table_t::value_type(
					table_t::key_type(key.begin(), key.end(), this->allocator())
					, table_t::mapped_type(cvalue, cvalue + value_len, this->allocator())
					)
				);
#endif
		}

		auto update_by_issue_41(
			const std::string &key,
			const void * value,
			const std::size_t value_len,
			void * /* old_value */,
			const std::size_t old_value_len
		) -> status_t
		{
		  /* hstore issue 41: "a put should replace any existing k,v pairs that match.
		   * If the new put is a different size, then the object should be reallocated.
		   * If the new put is the same size, then it should be updated in place."
		   */
			if ( value_len != old_value_len )
			{
				auto p_key = KEY_T(key.begin(), key.end(), this->allocator());
				return enter_replace(p_key, value, value_len);
			}
			else
			{
				std::vector<std::unique_ptr<Component::IKVStore::Operation>> v;
				v.emplace_back(std::make_unique<Component::IKVStore::Operation_write>(0, value_len, value));
				std::vector<Component::IKVStore::Operation *> v2;
				std::transform(v.begin(), v.end(), std::back_inserter(v2), [] (const auto &i) { return i.get(); });
				return this->atomic_update(key, v2);
			}
		}

		auto get(
			const std::string &key,
			void*& out_value,
			std::size_t& out_value_len
		) const -> status_t
		try
		{
			auto p_key = KEY_T(key.begin(), key.end(), this->allocator());
			auto &v = map().at(p_key);

			if ( out_value == nullptr )
			{
				out_value = ::malloc(v.size());
				if ( ! out_value )
				{
					throw std::bad_alloc();
				}
			}
			else
			{
				/* Although not documented, assume that non-zero
				 * out_value implies that out_value_len holds
				 * the buffer's size.
				 *
				 * It might be reaonable to
				 *  a) fill the buffer and/or
				 *  b) return the necessary size in out_value_len,
				 * but neither action is documented, so we do not.
				 */
				if ( out_value_len < v.size() )
				{
					return Component::IKVStore::E_INSUFFICIENT_BUFFER;
				}
			}

			out_value_len = v.size();
			memcpy(out_value, v.data(), out_value_len);
			return Component::IKVStore::S_OK;
		}
		catch ( std::out_of_range & )
		{
			return Component::IKVStore::E_KEY_NOT_FOUND;
		}

		auto get_direct(
			const std::string & key
			, void* out_value
			, std::size_t & out_value_len
		) const -> status_t
		try
		{
			auto p_key = KEY_T(key.begin(), key.end(), this->allocator());

			auto &v = this->map().at(p_key);

			auto value_len = v.size();
			if (out_value_len < value_len)
			{
				/* NOTE: it might be helpful to tell the caller how large
				 * a buffer is needed,
				 * but that does not seem to be expected.
				 */
				return Component::IKVStore::E_INSUFFICIENT_BUFFER;
			}

			out_value_len = value_len;

			assert(out_value);

			/* memcpy for moment
			*/
			memcpy(out_value, v.data(), out_value_len);
			return Component::IKVStore::S_OK;
		}
		catch ( const std::out_of_range & )
		{
		        return Component::IKVStore::E_KEY_NOT_FOUND;
		}

		auto lock(
			const std::string &key
			, hstore::lock_type_t type
			, void *& out_value
			, std::size_t & out_value_len
		) -> Component::IKVStore::key_t
		{
			const auto p_key = KEY_T(key.begin(), key.end(), this->allocator());

			try
			{
				MAPPED_T &val = this->map().at(p_key);
				if ( ! try_lock(this->map(), type, p_key) )
				{
					return Component::IKVStore::KEY_NONE;
				}
				out_value = val.data();
				out_value_len = val.size();
			}
			catch ( std::out_of_range & )
			{
				/* if the key is not found, we create it and
				* allocate value space equal in size to out_value_len
				*/
				if ( option_DEBUG )
				{
					PLOG(PREFIX "allocating object %lu bytes", __func__, out_value_len);
				}

				auto r =
					this->map().emplace(
						std::piecewise_construct
						, std::forward_as_tuple(p_key)
						, std::forward_as_tuple(out_value_len, this->allocator())
					);

				if ( ! r.second )
				{
					return Component::IKVStore::KEY_NONE;
				}

				out_value = r.first->second.data();
				out_value_len = r.first->second.size();
			}
			return new lock_impl(key);
		}

		auto unlock(Component::IKVStore::key_t key_) -> status_t
		{
			if ( key_ )
			{
				if ( auto lk = dynamic_cast<lock_impl *>(key_) )
				{
					try {
						auto p_key = KEY_T(lk->key().begin(), lk->key().end(), this->allocator());
						this->map().unlock(p_key);
					}
					catch ( const std::out_of_range &e )
					{
						return Component::IKVStore::E_KEY_NOT_FOUND;
					}
					catch( ... ) {
						throw General_exception(PREFIX "failed unexpectedly", __func__);
					}
					delete lk;
				}
				else
				{
					return Component::IKVStore::S_OK; /* not really OK - was not one of our locks */
				}
			}
			return Component::IKVStore::S_OK;
		}

		auto locate_apply(
			const KEY_T &p_key
			, std::size_t object_size
		) -> MAPPED_T *
		{
			try
			{
				return &this->map().at(p_key);
			}
			catch ( const std::out_of_range & )
			{
				/* if the key is not found, we create it and
				allocate value space equal in size to out_value_len
				*/

				if ( option_DEBUG )
				{
					PLOG(PREFIX "allocating object %lu bytes", __func__, object_size);
				}

				auto r =
					this->map().emplace(
						std::piecewise_construct
							, std::forward_as_tuple(p_key)
						, std::forward_as_tuple(object_size, this->allocator())
					);
				return
					r.second
					? &(*r.first).second
					: nullptr
					;
			}
		}

		auto apply(
			const std::string &key
			, std::function<void(void*,std::size_t)> functor
			, std::size_t object_size
		) -> status_t
		{
			auto p_key = KEY_T(key.begin(), key.end(), this->allocator());
			if ( auto val = locate_apply(p_key, object_size) )
			{
				auto data = static_cast<char *>(val->data());
				auto data_len = val->size();
				functor(data, data_len);
				return Component::IKVStore::S_OK;
			}
			else
			{
				return Component::IKVStore::E_KEY_NOT_FOUND;
			}
		}

		auto lock_and_apply(
			const std::string &key
			, std::function<void(void*,std::size_t)> functor
			, std::size_t object_size
		) -> status_t
		{
			auto p_key = KEY_T(key.begin(), key.end(), this->allocator());
			if ( auto val = locate_apply(p_key, object_size) )
			{
				auto data = static_cast<char *>(val->data());
				auto data_len = val->size();
				definite_lock m(this->map(), p_key);
				functor(data, data_len);
				return Component::IKVStore::S_OK;
			}
			else
			{
				return Component::IKVStore::E_KEY_NOT_FOUND;
			}
		}

		auto erase(
			const std::string &key
		) -> status_t
		{
			auto p_key = KEY_T(key.begin(), key.end(), this->allocator());
			return
				map().erase(p_key) == 0
				? Component::IKVStore::E_KEY_NOT_FOUND
				: Component::IKVStore::S_OK
			;
		}

		auto count() const -> std::size_t
		{
			return map().size();
		}

		auto bucket_count() const -> std::size_t
		{
			table_t::size_type count = 0;
			/* bucket counter */
			for (
				auto n = this->map().bucket_count()
				; n != 0
				; --n
			)
			{
				auto last = this->map().end(n-1);
				for ( auto first = this->map().begin(n-1); first != last; ++first )
				{
			                ++count;
				}
			}
			return count;
		}

		auto map(
			std::function
			<
				int(const std::string &key, const void *val, std::size_t val_len)
			> function
		) -> void
		{
			for ( auto &mt : this->map() )
			{
				const auto &pstring = mt.first;
				std::string s(static_cast<const char *>(pstring.data()), pstring.size());
				_functor(s, mt.second, &function);
			}

		}

		auto atomic_update_inner(
			KEY_T &key
			, const std::vector<Component::IKVStore::Operation *> &op_vector
		) -> status_t
		{
			return this->enter_update(key, op_vector.begin(), op_vector.end());
		}

		auto atomic_update(
			const std::string& key
			, const std::vector<Component::IKVStore::Operation *> &op_vector
		) -> status_t
		{
			auto p_key = KEY_T(key.begin(), key.end(), this->allocator());
			return this->atomic_update_inner(p_key, op_vector);
		}

		auto lock_and_atomic_update(
			const std::string& key
			, const std::vector<Component::IKVStore::Operation *> &op_vector
		) -> status_t
		{
			auto p_key = KEY_T(key.begin(), key.end(), this->allocator());
			definite_lock m(this->map(), p_key);
			return this->atomic_update_inner(p_key, op_vector);
		}
	};

#endif
