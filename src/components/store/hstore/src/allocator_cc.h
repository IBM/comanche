#ifndef COMANCHE_HSTORE_ALLOCATOR_H
#define COMANCHE_HSTORE_ALLOCATOR_H

#include "deallocator_cc.h"

#include "bad_alloc_cc.h"
#include "heap_cc.h"
#include "persister_cc.h"

#include <cstddef> /* size_t, ptrdiff_t */

template <typename T, typename Persister>
	class allocator_cc;

template <>
	class allocator_cc<void, persister>
		: public deallocator_cc<void, persister>
	{
	public:
		using deallocator_type = deallocator_cc<void, persister>;
		using typename deallocator_type::pointer;
		using typename deallocator_type::const_pointer;
		using typename deallocator_type::value_type;

		template <typename U>
			struct rebind
			{
				using other = allocator_cc<U, persister>;
			};
	};

template <typename Persister>
	class allocator_cc<void, Persister>
		: public deallocator_cc<void, Persister>
	{
	public:
		using deallocator_type = deallocator_cc<void, Persister>;
		using typename deallocator_type::pointer;
		using typename deallocator_type::const_pointer;
		using typename deallocator_type::value_type;
		template <typename U>
			struct rebind
			{
				using other = allocator_cc<U, Persister>;
			};
	};

template <typename T, typename Persister = persister>
	class allocator_cc
		: public deallocator_cc<T, Persister>
	{
	public:
		using deallocator_type = deallocator_cc<T, Persister>;
		using size_type = std::size_t;
		using difference_type = std::ptrdiff_t;
		using typename deallocator_type::pointer;
		using typename deallocator_type::const_pointer;
		using reference = T &;
		using const_reference = const T &;
		using typename deallocator_type::value_type; // = T;

		template <typename U>
			struct rebind
			{
				using other = allocator_cc<U, Persister>;
			};

		explicit allocator_cc(void *area_, std::size_t size_, Persister p_ = Persister())
			: deallocator_cc<T, Persister>(area_, size_, p_)
		{}

		explicit allocator_cc(void *area_, Persister p_ = Persister())
			: deallocator_cc<T, Persister>(area_, p_)
		{}

		allocator_cc(const heap_cc &pool_, Persister p_ = Persister()) noexcept
			: deallocator_cc<T, Persister>(pool_, (p_))
		{}

		allocator_cc(const allocator_cc &a_) noexcept = default;

		template <typename U, typename P>
			allocator_cc(const allocator_cc<U, P> &a_) noexcept
				: allocator_cc(a_.pool())
			{}

		allocator_cc &operator=(const allocator_cc &a_) = delete;

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
			, typename allocator_cc<void, Persister>::const_pointer /* hint */ =
				typename allocator_cc<void, Persister>::const_pointer{}
			, const char * = nullptr
		) -> pointer
		{
			auto ptr = this->pool().malloc(s * sizeof(T));
			if ( ptr == 0 )
			{
				throw bad_alloc_cc(0, s, sizeof(T));
			}
			return static_cast<pointer>(ptr);
		}
	};

#endif
