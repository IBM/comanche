/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef _COMANCHE_HSTORE_DEALLOCATOR_POBJ_H
#define _COMANCHE_HSTORE_DEALLOCATOR_POBJ_H

#include "persister_pmem.h"
#include "pointer_pobj.h"
#include "trace_flags.h"

#pragma GCC diagnostic push
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <libpmemobj.h> /* pmemobj_free */
#pragma GCC diagnostic pop

#include <cstdlib> /* size_t, ptrdiff_t */

#if TRACE_PALLOC || TRACE_PERSIST
#include <iostream> /* cerr */
#endif

template <typename T, typename Persister = persister_pmem>
	class deallocator_pobj;

template <>
	class deallocator_pobj<void, persister_pmem>
		: public persister_pmem
	{
	public:
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using pointer = pointer_pobj<void, 0U>;
		using const_pointer = pointer_pobj<const void, 0U>;
		using value_type = void;
		template <typename U, typename P = persister_pmem>
			struct rebind
			{
				using other = deallocator_pobj<U, P>;
			};
	};

template <typename T, typename Persister>
	class deallocator_pobj
		: public Persister
	{
	public:
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using pointer = pointer_pobj<T, 0U>;
		using const_pointer = pointer_pobj<const T, 0U>;
		using reference = T &;
		using const_reference = const T &;
		using value_type = T;

		template <typename U>
			struct rebind
			{
				using other = deallocator_pobj<U, Persister>;
			};

		deallocator_pobj() noexcept
		{}

		deallocator_pobj(const deallocator_pobj &) noexcept
		{}

		template <typename U>
			deallocator_pobj(const deallocator_pobj<U, Persister> &) noexcept
				: deallocator_pobj()
			{}

		deallocator_pobj &operator=(const deallocator_pobj &) = delete;

		pointer address(reference x) const noexcept
		{
			return pointer(pmemobj_oid(&x));
		}
		const_pointer address(const_reference x) const noexcept
		{
			return pointer(pmemobj_oid(&x));
		}

		void deallocate(
			pointer oid
			, size_type
#if TRACE_PALLOC
				s
#endif
		)
		{
#if TRACE_PALLOC
			{
				auto ptr = static_cast<char *>(pmemobj_direct(oid));
				std::cerr << __func__
					<< " [" << ptr
					<< ".." << static_cast<void *>(ptr + s * sizeof(T))
					<< ")\n";
			}
#endif
			pmemobj_free(&oid);
		}
		auto max_size() const
		{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
			return PMEMOBJ_MAX_ALLOC_SIZE;
#pragma GCC diagnostic pop
		}
		void persist(const void *ptr, size_type len
			, const char *
#if TRACE_PERSIST
			what
#endif
				= "unspecified"
		) const
		{
#if TRACE_PERSIST
			std::cerr << __func__ << " " << what << " ["
				<< ptr << ".."
				<< static_cast<const void *>(static_cast<const char*>(ptr)+len)
				<< ")\n";
#endif
			Persister::persist(ptr, len);
		}
	};

#endif
