#ifndef _DAWN_POBJ_ALLOCATOR_H
#define _DAWN_POBJ_ALLOCATOR_H

#include "pobj_pointer.h"

#include "persister.h"
#include "pobj_bad_alloc.h"
#include "trace_flags.h"

#pragma GCC diagnostic push
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <libpmemobj.h>
#include <libpmem.h> /* pmem_persist */
#pragma GCC diagnostic pop

#include <cerrno>
#include <cstdlib> /* size_t, ptrdiff_t */
#include <cstdint> /* uint64_t */

#if TRACE_PALLOC
#include <iostream> /* cerr */
#endif

template <typename T>
	class pobj_allocator;

template <>
	class pobj_allocator<void>
	{
	public:
		using pointer = pobj_pointer<void>;
		using const_pointer = pobj_pointer<const void>;
		using value_type = void;
		template <typename U>
			struct rebind
			{
				using other = pobj_allocator<U>;
			};
	};

template <typename T>
	class pobj_allocator
		: public persister
	{
		PMEMobjpool * _pool;
		std::uint64_t _type_num;
	public:
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using pointer = pobj_pointer<T>;
		using const_pointer = pobj_pointer<const T>;
		using reference = T &;
		using const_reference = const T &;
		using value_type = T;
		template <typename U>
			struct rebind
			{
				using other = pobj_allocator<U>;
			};

		pobj_allocator(PMEMobjpool * pool_, std::uint64_t type_num_) noexcept
			: _pool(pool_)
			, _type_num(type_num_)
		{}

		pobj_allocator(const pobj_allocator &a_) noexcept
			: pobj_allocator(a_.pool(), a_.type_num())
		{}

		template <typename U>
			pobj_allocator(const pobj_allocator<U> &a_) noexcept
				: pobj_allocator(a_.pool(), a_.type_num())
			{}

		pobj_allocator &operator=(const pobj_allocator &a_) = delete;

		pointer address(reference x) const noexcept
		{
			return pointer(pmemobj_oid(&x));
		}
		const_pointer address(const_reference x) const noexcept
		{
			return pointer(pmemobj_oid(&x));
		}

		auto allocate(
			size_type s
			, pobj_allocator<void>::const_pointer /* hint */ =
					pobj_allocator<void>::const_pointer{}
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
				throw pobj_bad_alloc(errno);
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
			return pobj_pointer<T>(oid);
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
		void persist(const void *ptr, size_type len, const char *
#if TRACE_PALLOC
			what
#endif
		) const override
	{
#if TRACE_PALLOC
		std::cerr << __func__ << " " << what << " ["
			<< ptr << ".."
			<< static_cast<const void *>(static_cast<const char*>(ptr)+len)
			<< ")\n";
#endif
		pmem_persist(ptr, len);
	}
	PMEMobjpool *pool() const
	{
		return _pool;
	}
	std::uint64_t type_num() const
	{
		return _type_num;
	}
};

template <typename T>
	class pobj_cache_aligned_allocator;

template <>
	class pobj_cache_aligned_allocator<void>
	{
	public:
		using pointer = PMEMoid *;
		using const_pointer = PMEMoid *;
		using value_type = void;
		template <typename U>
			struct rebind
			{
				using other = pobj_cache_aligned_allocator<U>;
			};
	};

template <typename T>
	class pobj_cache_aligned_allocator
		: private pobj_allocator<T>
	{
		using base = pobj_allocator<T>;
		static constexpr std::size_t cache_align = 48U;
	public:
		using typename base::size_type;
		using typename base::difference_type;
		using typename base::pointer;
		using typename base::const_pointer;
		using typename base::reference;
		using typename base::const_reference;
		using typename base::value_type;
		using base::pool;
		using base::type_num;
		using base::persist;
		using base::address;
		template <typename U>
			struct rebind
			{
				using other = pobj_cache_aligned_allocator<U>;
			};
		pobj_cache_aligned_allocator(
			PMEMobjpool * pool_
			, std::uint64_t type_num_
		) noexcept
			: base(pool_, type_num_)
		{}

		pobj_cache_aligned_allocator(
			const pobj_cache_aligned_allocator &a_
		) noexcept
			: pobj_cache_aligned_allocator(a_.pool(), a_.type_num())
		{}

		template <typename U>
			pobj_cache_aligned_allocator(
				const pobj_cache_aligned_allocator<U> &a_
			) noexcept
				: pobj_cache_aligned_allocator(a_.pool(), a_.type_num())
			{}

		pobj_cache_aligned_allocator &operator=(
			const pobj_cache_aligned_allocator &a_
		) = delete;

		auto allocate(
			size_type s
			, pobj_cache_aligned_allocator<void>::const_pointer /* hint */ =
				pobj_cache_aligned_allocator<void>::const_pointer{}
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
				throw pobj_bad_alloc(errno);
			}
#if TRACE_PALLOC
			{
				auto ptr = static_cast<char *>(pmemobj_direct(oid)) + cache_align;
				std::cerr << __func__
					<< " " << (why ? why : "(cache aligned no reason)")
					<< " [" << static_cast<void *>(ptr)
					<< ".." << static_cast<void *>(ptr + s * sizeof(T))
					<< ")\n";
			}
#endif
			return pointer(oid);
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
				auto ptr = static_cast<char *>(pmemobj_direct(oid)) - cache_align;
				std::cerr << __func__
					<< " [" << static_cast<void *>(ptr)
					<< ".." << static_cast<void *>(ptr + s * sizeof(T))
					<< ")\n"
					;
			}
#endif
			pmemobj_free(&oid);
		}
	};

#endif
