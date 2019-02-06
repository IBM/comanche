#ifndef _COMANCHE_DEALLOCATOR_POBJ_CACHE_ALIGNED_H
#define _COMANCHE_DEALLOCATOR_POBJ_CACHE_ALIGNED_H

#include "deallocator_pobj.h"

#include "pointer_pobj.h"
#include "persister_pmem.h"
#include "pobj_bad_alloc.h"
#include "trace_flags.h"

#pragma GCC diagnostic push
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <libpmemobj.h>
#pragma GCC diagnostic pop

#include <cstdlib> /* size_t, ptrdiff_t */

#if TRACE_PALLOC
#include <iostream> /* cerr */
#endif

template <typename T, typename Deallocator = deallocator_pobj<T>>
	class deallocator_pobj_cache_aligned;

template <>
	class deallocator_pobj_cache_aligned<void, deallocator_pobj<void>>
	{
	protected:
		static unsigned constexpr cache_align = 48U;
	public:
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using pointer = pointer_pobj<void, cache_align>;
		using const_pointer = pointer_pobj<const void, cache_align>;
		using value_type = void;
		template <typename U>
			struct rebind
			{
				using other = deallocator_pobj_cache_aligned<U>;
			};
	};

template <typename T, typename Deallocator>
	class deallocator_pobj_cache_aligned
		: public Deallocator
	{
	protected:
		static unsigned constexpr cache_align = 48U;
	public:
		using typename Deallocator::value_type;
		using typename Deallocator::size_type;
		using typename Deallocator::difference_type;
		using pointer = pointer_pobj<value_type, cache_align>;
		using const_pointer = pointer_pobj<const value_type, cache_align>;
		using typename Deallocator::reference;
		using typename Deallocator::const_reference;
		using Deallocator::persist;
		pointer address(reference x) const noexcept
		{
			return pointer(pmemobj_oid(static_cast<char *>(static_cast<void *>(&x)) - cache_align));
		}
		const_pointer address(const_reference x) const noexcept
		{
			return pointer(pmemobj_oid(static_cast<const char *>(static_cast<const void *>(&x)) - cache_align));
		}

		template <typename U>
			struct rebind
			{
				using other = deallocator_pobj_cache_aligned<U>;
			};

		deallocator_pobj_cache_aligned() noexcept = default; // {}

		deallocator_pobj_cache_aligned(
			const deallocator_pobj_cache_aligned &
		) noexcept = default; // {}

		template <typename D>
			deallocator_pobj_cache_aligned(
				const deallocator_pobj_cache_aligned<D> &
			) noexcept
				: deallocator_pobj_cache_aligned()
			{}

		deallocator_pobj_cache_aligned &operator=(
			const deallocator_pobj_cache_aligned &
		) = delete;

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
				auto ptr = static_cast<char *>(pmemobj_direct(oid)) - cache_align;
				std::cerr << __func__
					<< " [" << static_cast<void *>(ptr)
					<< ".." << static_cast<void *>(ptr + s * sizeof(value_type))
					<< ") OID " << std::hex << oid.pool_uuid_lo << "." << oid.off << std::dec << "\n"
					;
			}
#endif
			pmemobj_free(&oid);
		}
	};

#endif
