#ifndef _COMANCHE_POINTER_POBJ_H
#define _COMANCHE_POINTER_POBJ_H

#pragma GCC diagnostic push
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <libpmemobj.h> /* PMEMoid, pmemobj_direct, OID_IS_NULL */
#pragma GCC diagnostic pop

#include <cstdlib> /* ptrdiff_t */

template <typename T, unsigned Offset>
	class pointer_pobj;

template <unsigned Offset>
	class pointer_pobj<void, Offset>
		: public PMEMoid
	{
		static_assert(Offset == 48U, "lost offset");
	public:
		explicit pointer_pobj() noexcept
			: PMEMoid()
		{}
		explicit pointer_pobj(const PMEMoid &oid) noexcept
			: PMEMoid(oid)
		{}
		pointer_pobj(const pointer_pobj &) = default;
		pointer_pobj &operator=(const pointer_pobj &) = default;
	};

template <unsigned Offset>
	class pointer_pobj<const void, Offset>
		: public PMEMoid
	{
		static_assert(Offset == 48U, "lost offset");
	public:
		explicit pointer_pobj() noexcept
			: PMEMoid()
		{}
		explicit pointer_pobj(const PMEMoid &oid) noexcept
			: PMEMoid(oid)
		{}
		pointer_pobj(const pointer_pobj &) = default;
		pointer_pobj &operator=(const pointer_pobj &) = default;
	};

template <typename T, unsigned Offset>
	class pointer_pobj
		: public PMEMoid
	{
		static_assert(Offset == 48U, "lost offset");
		const void *offset_address() const noexcept
		{
			return static_cast<const char *>(pmemobj_direct(*this)) + Offset;
		}
	public:
		using element_type = T;
		using difference_type = std::ptrdiff_t;
		template <typename U>
			struct rebind
			{
				using other = pointer_pobj<U, Offset>;
			};

		explicit pointer_pobj() noexcept
			: PMEMoid()
		{}
		explicit pointer_pobj(const PMEMoid &oid) noexcept
			: PMEMoid(oid)
		{}
		pointer_pobj(const pointer_pobj &) = default;
		pointer_pobj &operator=(const pointer_pobj &) = default;
		T &operator*() const noexcept
		{
			return *const_cast<T *>(
				static_cast<const T *>(
					offset_address()
				)
			);
		}
		T *operator->() const noexcept
		{
			return const_cast<T *>(
				static_cast<const T *>(
					offset_address()
				)
			);
		}
		operator bool() const { return ! OID_IS_NULL(*this); }
	};

#endif
