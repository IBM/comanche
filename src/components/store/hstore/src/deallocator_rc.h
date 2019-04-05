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


#ifndef COMANCHE_HSTORE_DEALLOCATOR_RC_H
#define COMANCHE_HSTORE_DEALLOCATOR_RC_H

#include "heap_rc.h"
#include "persister_cc.h"

#include <cstddef> /* size_t, ptrdiff_t */

template <typename T, typename Persister>
	class deallocator_rc;

template <>
	class deallocator_rc<void, persister>
	{
	public:
		using pointer = void *;
		using const_pointer = const void *;
		using value_type = void;
		template <typename U>
			struct rebind
			{
				using other = deallocator_rc<U, persister>;
			};
	};

template <typename Persister>
	class deallocator_rc<void, Persister>
	{
	public:
		using pointer = void *;
		using const_pointer = const void *;
		using value_type = void;
		template <typename U>
			struct rebind
			{
				using other = deallocator_rc<U, Persister>;
			};
	};

template <typename T, typename Persister = persister>
	class deallocator_rc
		: public Persister
	{
		heap_rc _pool;
	public:
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using pointer = T*;
		using const_pointer = const T*;
		using reference = T &;
		using const_reference = const T &;
		using value_type = T;

	template <typename U>
		struct rebind
		{
			using other = deallocator_rc<U, Persister>;
		};

		explicit deallocator_rc(void *area_, std::size_t size_, Persister p_ = Persister())
			: Persister(p_)
			, _pool(area_, size_)
		{}

		explicit deallocator_rc(void *area_, Persister p_ = Persister())
			: Persister(p_)
			, _pool(area_)
		{}

		explicit deallocator_rc(const heap_rc &pool_, Persister p_ = Persister()) noexcept
			: Persister(p_)
			, _pool(pool_)
		{}

		explicit deallocator_rc(const deallocator_rc &) noexcept = default;

		template <typename U, typename P>
			explicit deallocator_rc(const deallocator_rc<U, P> &d_) noexcept
				: deallocator_rc(d_.pool())
			{}

		deallocator_rc &operator=(const deallocator_rc &e_) = delete;

		pointer address(reference x) const noexcept
		{
			return pointer(&x);
		}
		const_pointer address(const_reference x) const noexcept
		{
			return pointer(&x);
		}

		void deallocate(
			pointer p
			, size_type sz_
		)
		{
			_pool.free(p, sz_);
		}

		auto max_size() const
		{
			return 8; /* reminder to provide a proper max size value */
		}

		void persist(const void *ptr, size_type len, const char * = nullptr)
		{
			Persister::persist(ptr, len);
		}

		auto pool() const
		{
			return _pool;
		}
	};

#endif
