#ifndef COMANCHE_HSTORE_DEALLOCATOR_H
#define COMANCHE_HSTORE_DEALLOCATOR_H

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
			, size_type
		)
		{
			_pool.free(p);
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
