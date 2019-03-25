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


#ifndef _COMANCHE_HSTORE_ALLOCATOR_POBJ_CACHE_ALIGNED_H
#define _COMANCHE_HSTORE_ALLOCATOR_POBJ_CACHE_ALIGNED_H

#include "deallocator_pobj_cache_aligned.h"
#include "pool_pobj.h"

#include "persister_pmem.h"
#include "pobj_bad_alloc.h"
#include "pointer_pobj.h"
#include "trace_flags.h"

#pragma GCC diagnostic push
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <libpmemobj.h>
#pragma GCC diagnostic pop

#include <cerrno>
#include <cstdlib> /* size_t, ptrdiff_t */
#include <cstdint> /* uint64_t */

#if TRACE_PALLOC
#include <iostream> /* cerr */
#endif

template <typename T>
	struct type_number;

template <typename T, typename Deallocator = deallocator_pobj_cache_aligned<T>>
	class allocator_pobj_cache_aligned;

template <>
	class allocator_pobj_cache_aligned<void, deallocator_pobj_cache_aligned<void>>
		: public deallocator_pobj_cache_aligned<void>
	{
		using deallocator_type = deallocator_pobj_cache_aligned<void>;
	protected:
		using deallocator_type::cache_align;
	public:
		using typename deallocator_type::pointer;
		using typename deallocator_type::const_pointer;
		using typename deallocator_type::value_type;
		template <typename U>
			struct rebind
			{
				using other =
					allocator_pobj_cache_aligned<U>;
			};
	};

template <typename T, typename Deallocator>
	class allocator_pobj_cache_aligned
		: public Deallocator
		, public pool_pobj
	{
	public:
		using deallocator_type = Deallocator;
	protected:
		using deallocator_type::cache_align;
	public:
		using typename deallocator_type::size_type;
		using typename deallocator_type::difference_type;
		using typename deallocator_type::pointer;
		using typename deallocator_type::const_pointer;
		using typename deallocator_type::reference;
		using typename deallocator_type::const_reference;
		using typename deallocator_type::value_type;
		using deallocator_type::address;
		using deallocator_type::persist;
		static constexpr std::uint64_t type_num() { return type_number<T>::value; }

		template <typename U>
			struct rebind
			{
				using other = allocator_pobj_cache_aligned<U>;
			};
		allocator_pobj_cache_aligned(PMEMobjpool * pool_) noexcept
			: pool_pobj(pool_)
		{}

		allocator_pobj_cache_aligned(const allocator_pobj_cache_aligned &a_) noexcept
			: allocator_pobj_cache_aligned(a_.pool())
		{}

		template <typename U>
			allocator_pobj_cache_aligned(
				const allocator_pobj_cache_aligned<U> &a_
			) noexcept
				: allocator_pobj_cache_aligned(a_.pool())
			{}

		allocator_pobj_cache_aligned &operator=(
			const allocator_pobj_cache_aligned &a_
		) = delete;

		auto allocate(
			size_type s
			, allocator_pobj_cache_aligned<void>::const_pointer /* hint */ =
				allocator_pobj_cache_aligned<void>::const_pointer{}
			, const char *
#if TRACE_PALLOC
			why
#endif
				= nullptr
		) -> pointer
		{
			PMEMoid oid;
			auto r =
				pmemobj_alloc(
					this->pool(), &oid
					, cache_align + s * sizeof(T)
					, this->type_num()
					, nullptr
					, nullptr
				);
			if ( r != 0 )
			{
				throw pobj_bad_alloc(cache_align, s, sizeof(T), sizeof(T), errno);
			}
#if TRACE_PALLOC
			{
				auto ptr = static_cast<char *>(pmemobj_direct(oid)) + cache_align;
				std::cerr << __func__
					<< " " << (why ? why : "(cache aligned no reason)")
					<< " [" << static_cast<void *>(ptr)
					<< ".." << static_cast<void *>(ptr + s * sizeof(T))
					<< ") OID " << std::hex << oid.pool_uuid_lo << "." << oid.off << "\n";
			}
#endif
			return pointer(oid);
		}
	};

#endif
