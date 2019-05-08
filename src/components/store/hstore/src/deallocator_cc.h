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


#ifndef COMANCHE_HSTORE_DEALLOCATOR_CC_H
#define COMANCHE_HSTORE_DEALLOCATOR_CC_H

#include "heap_cc.h"
#include "persister_cc.h"

#include <cstddef> /* size_t, ptrdiff_t */

template <typename T, typename Persister>
	class deallocator_cc;

template <>
	class deallocator_cc<void, persister>
	{
	public:
		using pointer = void *;
		using const_pointer = const void *;
		using value_type = void;
		template <typename U>
			struct rebind
			{
				using other = deallocator_cc<U, persister>;
			};
	};

template <typename Persister>
	class deallocator_cc<void, Persister>
	{
	public:
		using pointer = void *;
		using const_pointer = const void *;
		using value_type = void;
		template <typename U>
			struct rebind
			{
				using other = deallocator_cc<U, Persister>;
			};
	};

template <typename T, typename Persister = persister>
	class deallocator_cc
		: public Persister
	{
		heap_cc _pool;
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
			using other = deallocator_cc<U, Persister>;
		};

		explicit deallocator_cc(void *area_, std::size_t size_, Persister p_ = Persister())
			: Persister(p_)
			, _pool(area_, size_)
		{}

		explicit deallocator_cc(void *area_, Persister p_ = Persister())
			: Persister(p_)
			, _pool(area_)
		{}

		explicit deallocator_cc(const heap_cc &pool_, Persister p_ = Persister()) noexcept
			: Persister(p_)
			, _pool(pool_)
		{}

		explicit deallocator_cc(const deallocator_cc &) noexcept = default;

		template <typename U, typename P>
			explicit deallocator_cc(const deallocator_cc<U, P> &d_) noexcept
				: deallocator_cc(d_.pool())
			{}

		deallocator_cc &operator=(const deallocator_cc &e_) = delete;

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
			_pool.free(p, sizeof(T) * sz_);
		}

		auto max_size() const
		{
			return 8; /* reminder to provide a proper max size value */
		}

		void persist(const void *ptr, size_type len, const char * = nullptr) const
		{
			Persister::persist(ptr, len);
		}

		auto pool() const
		{
			return _pool;
		}
	};

#endif
