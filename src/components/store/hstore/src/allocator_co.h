#ifndef COMANCHE_HSTORE_ALLOCATOR_CO_H
#define COMANCHE_HSTORE_ALLOCATOR_CO_H

#include "deallocator_co.h"

#include "bad_alloc_cc.h"
#include "persister_cc.h"

#include <cstddef> /* size_t, ptrdiff_t */
#include <new> /* bad_alloc */

template <typename T, typename Persister>
	class allocator_co;

template <>
	class allocator_co<void, persister>
		: public deallocator_co<void, persister>
	{
	public:
		using deallocator_type = deallocator_co<void, persister>;
		using typename deallocator_type::pointer;
		using typename deallocator_type::const_pointer;
		using typename deallocator_type::value_type;
		template <typename U>
			struct rebind
			{
				using other = allocator_co<U, persister>;
			};
	};

template <typename Persister>
	class allocator_co<void, Persister>
		: public deallocator_co<void, Persister>
	{
	public:
		using deallocator_type = deallocator_co<void, Persister>;
		using typename deallocator_type::pointer;
		using typename deallocator_type::const_pointer;
		using typename deallocator_type::value_type;
		template <typename U>
			struct rebind
			{
				using other = allocator_co<U, Persister>;
			};
	};

template <typename T, typename Persister = persister>
	class allocator_co
		: public deallocator_co<T, Persister>
	{
		heap_co *_heap;
	public:
		using deallocator_type = deallocator_co<T, Persister>;
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
				using other = allocator_co<U, Persister>;
			};

		explicit allocator_co(heap_co &heap_, Persister p_ = Persister())
			: deallocator_co<T, Persister>(p_)
			, _heap(&heap_)
		{}

		allocator_co(const allocator_co &a_) noexcept = default;

		template <typename U, typename P>
			allocator_co(const allocator_co<U, P> &a_) noexcept
				: allocator_co(a_.heap())
			{}

		allocator_co &operator=(const allocator_co &a_) = delete;

#if 0
		/* deprecated in C++20 */
		pointer address(reference x) const noexcept
		{
			return pointer(&x);
		}
		const_pointer address(const_reference x) const noexcept
		{
			return pointer(&x);
		}
#endif

		auto allocate(
			size_type s
			, typename allocator_co<void, Persister>::const_pointer /* hint */ =
					typename allocator_co<void, Persister>::const_pointer{}
			, const char * = nullptr
		) -> pointer
		{
			auto oid = heap().malloc(s * sizeof(T));
			if ( OID_IS_NULL(oid) )
			{
				throw bad_alloc_cc(0, s, sizeof(T));
			}
			return pointer(oid);
		}
		auto max_size() const
		{
			return 8; /* ERROR: provide a proper max size value */
		}
		void persist(const void *ptr, size_type len, const char * = nullptr)
		{
			Persister::persist(ptr, len);
		}
		auto &heap() const
		{
			return *_heap;
		}
	};

#endif
