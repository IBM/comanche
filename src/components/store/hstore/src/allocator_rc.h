/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef COMANCHE_HSTORE_ALLOCATOR_RC_H
#define COMANCHE_HSTORE_ALLOCATOR_RC_H

#include "deallocator_rc.h"

#include "bad_alloc_cc.h"
#include "persister_cc.h"

#include <cstddef> /* size_t, ptrdiff_t */

template <typename T, typename Persister>
	class allocator_rc;

template <>
	class allocator_rc<void, persister>
		: public deallocator_rc<void, persister>
	{
	public:
		using deallocator_type = deallocator_rc<void, persister>;
		using typename deallocator_type::pointer;
		using typename deallocator_type::const_pointer;
		using typename deallocator_type::value_type;

		template <typename U>
			struct rebind
			{
				using other = allocator_rc<U, persister>;
			};
	};

template <typename Persister>
	class allocator_rc<void, Persister>
		: public deallocator_rc<void, Persister>
	{
	public:
		using deallocator_type = deallocator_rc<void, Persister>;
		using typename deallocator_type::pointer;
		using typename deallocator_type::const_pointer;
		using typename deallocator_type::value_type;
		template <typename U>
			struct rebind
			{
				using other = allocator_rc<U, Persister>;
			};
	};

template <typename T, typename Persister = persister>
	class allocator_rc
		: public deallocator_rc<T, Persister>
	{
	public:
		using deallocator_type = deallocator_rc<T, Persister>;
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
				using other = allocator_rc<U, Persister>;
			};

		explicit allocator_rc(void *area_, std::size_t size_, Persister p_ = Persister())
			: deallocator_rc<T, Persister>(area_, size_, p_)
		{}

		explicit allocator_rc(void *area_, Persister p_ = Persister())
			: deallocator_rc<T, Persister>(area_, p_)
		{}

		allocator_rc(const heap_rc &pool_, Persister p_ = Persister()) noexcept
			: deallocator_rc<T, Persister>(pool_, (p_))
		{}

		allocator_rc(const allocator_rc &a_) noexcept = default;

		template <typename U, typename P>
			allocator_rc(const allocator_rc<U, P> &a_) noexcept
				: allocator_rc(a_.pool())
			{}

		allocator_rc &operator=(const allocator_rc &a_) = delete;

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
			, typename allocator_rc<void, Persister>::const_pointer /* hint */ =
				typename allocator_rc<void, Persister>::const_pointer{}
			, const char * = nullptr
		) -> pointer
		{
			auto ptr = this->pool().alloc(s * sizeof(T));
			if ( ptr == 0 )
			{
				throw bad_alloc_cc(0, s, sizeof(T));
			}
			return static_cast<pointer>(ptr);
		}

		void reconstitute(
			size_type s
			, typename allocator_rc<void, Persister>::const_pointer location
			, const char * = nullptr
		)
		{
			this->pool().inject_allocation(location, s * sizeof(T));
		}

		bool is_reconstituted(
			typename allocator_rc<void, Persister>::const_pointer location
		)
		{
			return this->pool().is_reconstituted(location);
		}
	};

#endif
