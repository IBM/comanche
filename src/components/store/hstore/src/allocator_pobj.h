#ifndef _COMANCHE_ALLOCATOR_POBJ_H
#define _COMANCHE_ALLOCATOR_POBJ_H

#include "deallocator_pobj.h"
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

template <typename T, typename Deallocator = deallocator_pobj<T>>
	class allocator_pobj;

template <>
	class allocator_pobj<void>
	{
	public:
		using pointer = pointer_pobj<void, 0U>;
		using const_pointer = pointer_pobj<const void, 0U>;
		using value_type = void;
		template <typename U, typename Persister>
			struct rebind
			{
				using other = allocator_pobj<U, Persister>;
			};
	};

template <typename T, typename Deallocator>
	class allocator_pobj
		: public Deallocator
		, public pool_pobj
	{
		using deallocator_type = Deallocator;
	public:
		using typename deallocator_type::size_type;
		using typename deallocator_type::difference_type;
		using typename deallocator_type::pointer;
		using typename deallocator_type::const_pointer;
		using typename deallocator_type::reference;
		using typename deallocator_type::const_reference;
		using typename deallocator_type::value_type;
		template <typename U>
			struct rebind
			{
				using other = allocator_pobj<U, Deallocator>;
			};

		allocator_pobj(PMEMobjpool * pool_, std::uint64_t type_num_) noexcept
			: pool_pobj(pool_, type_num_)
		{}

		allocator_pobj(const allocator_pobj &a_) noexcept
			: allocator_pobj(a_.pool(), a_.type_num())
		{}

		template <typename U>
			allocator_pobj(const allocator_pobj<U> &a_) noexcept
				: allocator_pobj(a_.pool(), a_.type_num())
			{}

		allocator_pobj &operator=(const allocator_pobj &a_) = delete;

		auto allocate(
			size_type s
			, allocator_pobj<void>::const_pointer /* hint */ =
					allocator_pobj<void>::const_pointer{}
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
					pool()
					, &oid
					, s * sizeof(T)
					, type_num()
					, nullptr
					, nullptr
				);
			if ( r != 0 )
			{
				throw 0, pobj_bad_alloc(s, sizeof(T), errno);
			}
#if TRACE_PALLOC
			{
				auto ptr = static_cast<char *>(pmemobj_direct(oid));
				std::cerr << __func__
					<< " " << (why ? why : "(unaligned no reason)")
					<< " [" << ptr
					<< ".." << static_cast<void *>(ptr + s * sizeof(T))
					<< ")\n";
			}
#endif
			return pointer_pobj<T, 0U>(oid);
		}
	};


#endif
