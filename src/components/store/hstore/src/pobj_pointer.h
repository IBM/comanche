#ifndef _DAWN_POBJ_POINTER_H
#define _DAWN_POBJ_POINTER_H

#pragma GCC diagnostic push
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <libpmemobj.h> /* PMEMoid, pmemobj_direct, OID_IS_NULL */
#pragma GCC diagnostic pop

#include <cstdlib> /* ptrdiff_t */

template <typename T>
	class pobj_pointer;

template <>
	class pobj_pointer<void>
		: public PMEMoid
	{
	public:
		explicit pobj_pointer() noexcept
			: PMEMoid()
		{}
		explicit pobj_pointer(const PMEMoid &oid) noexcept
			: PMEMoid(oid)
		{}
		pobj_pointer(const pobj_pointer &) = default;
		pobj_pointer &operator=(const pobj_pointer &) = default;
	};

template <>
	class pobj_pointer<const void>
		: public PMEMoid
	{
	public:
		explicit pobj_pointer() noexcept
			: PMEMoid()
		{}
		explicit pobj_pointer(const PMEMoid &oid) noexcept
			: PMEMoid(oid)
		{}
		pobj_pointer(const pobj_pointer &) = default;
		pobj_pointer &operator=(const pobj_pointer &) = default;
	};

template <typename T>
	class pobj_pointer
		: public PMEMoid
	{
	public:
		using element_type = T;
		using difference_type = std::ptrdiff_t;
		template <typename U>
			struct rebind
			{
				using other = pobj_pointer<U>;
			};

		explicit pobj_pointer() noexcept
			: PMEMoid()
		{}
		explicit pobj_pointer(const PMEMoid &oid) noexcept
			: PMEMoid(oid)
		{}
		pobj_pointer(const pobj_pointer &) = default;
		pobj_pointer &operator=(const pobj_pointer &) = default;
		T &operator*() const noexcept
		{
			return *static_cast<T *>(pmemobj_direct(*this));
		}
		T *operator->() const noexcept
		{
			return static_cast<T *>(pmemobj_direct(*this));
		}
		operator bool() const { return ! OID_IS_NULL(*this); }
	};

#endif
