#ifndef COMANCHE_HSTORE_DEALLOCATOR_CO_H
#define COMANCHE_HSTORE_DEALLOCATOR_CO_H

#include "heap_co.h"
#include "persister_pmem.h"
#include "pointer_pobj.h"
#include "store_root.h"
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
	class deallocator_co;

template <>
	class deallocator_co<void, persister_pmem>
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
				using other = deallocator_co<U, P>;
			};
	};

template <class Persister>
	class deallocator_co<void, Persister>
		: public Persister
	{
	public:
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using pointer = pointer_pobj<void, 0U>;
		using const_pointer = pointer_pobj<const void, 0U>;
		using value_type = void;
		template <typename U, typename P = Persister>
			struct rebind
			{
				using other = deallocator_co<U, P>;
			};
	};

template <typename T, typename Persister>
	class deallocator_co
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
				using other = deallocator_co<U, Persister>;
			};

		deallocator_co(const Persister = Persister()) noexcept
		{}

		deallocator_co(const deallocator_co &) noexcept = default;

		template <typename U>
			deallocator_co(const deallocator_co<U, Persister> &) noexcept
				: deallocator_co()
			{}

		deallocator_co &operator=(const deallocator_co &) = delete;

		pointer address(reference x) const noexcept /* to be deprecated */
		{
			return pointer(pmemobj_oid(&x));
		}
		const_pointer address(const_reference x) const noexcept /* to be deprecated */
		{
			return pointer(pmemobj_oid(&x));
		}

		void deallocate(
			pointer ptr
			, size_type
		)
		{
			auto pool = ::pmemobj_pool_by_oid(ptr);

			TOID_DECLARE_ROOT(struct store_root_t);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
			TOID(struct store_root_t) root = POBJ_ROOT(pool, struct store_root_t);
			assert(!TOID_IS_NULL(root));
#pragma GCC diagnostic ignored "-Wpedantic"
			auto heap = static_cast<heap_co *>(pmemobj_direct((D_RO(root)->heap_oid)));
#pragma GCC diagnostic pop
#if TRACE_PALLOC
			{
				auto ptr = static_cast<char *>(pmemobj_direct(oid));
				std::cerr << __func__
					<< " [" << ptr
					<< ".." << static_cast<void *>(ptr + s * sizeof(T))
					<< ")\n";
			}
#endif
			heap->free(ptr);
		}
		auto max_size() const
		{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
			return PMEMOBJ_MAX_ALLOC_SIZE;
#pragma GCC diagnostic pop
		}
		void persist(const void *ptr, size_type len, const char * = nullptr)
		{
			Persister::persist(ptr, len);
		}
	};

#endif
