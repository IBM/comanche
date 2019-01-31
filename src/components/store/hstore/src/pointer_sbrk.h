#ifndef _COMANCHE_POINTER_SBRK_H
#define _COMANCHE_POINTER_SBRK_H

#pragma GCC diagnostic push
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <libpmemobj.h> /* PMEMoid, pmemobj_direct, OID_IS_NULL */
#pragma GCC diagnostic pop

#include <cstdlib> /* ptrdiff_t */

class sbrk_addr
{
  char *location;
  std::uint64_t offset;
};

template <typename T, unsigned Offset>
	class pointer_sbrk;

template <unsigned Offset>
	class pointer_sbrk<void, Offset>
		: public PMEMoid
	{
	public:
		explicit pointer_sbrk() noexcept
			: PMEMoid()
		{}
		explicit pointer_sbrk(const PMEMoid &oid) noexcept
			: PMEMoid(oid)
		{}
		pointer_sbrk(const pointer_sbrk &) = default;
		pointer_sbrk &operator=(const pointer_sbrk &) = default;
	};

template <unsigned Offset>
	class pointer_sbrk<const void, Offset>
		: public PMEMoid
	{
	public:
		explicit pointer_sbrk() noexcept
			: PMEMoid()
		{}
		explicit pointer_sbrk(const PMEMoid &oid) noexcept
			: PMEMoid(oid)
		{}
		pointer_sbrk(const pointer_sbrk &) = default;
		pointer_sbrk &operator=(const pointer_sbrk &) = default;
	};

template <typename T, unsigned Offset>
	class pointer_sbrk
		: public PMEMoid
	{
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
				using other = pointer_sbrk<U, Offset>;
			};

		explicit pointer_sbrk() noexcept
			: PMEMoid()
		{}
		explicit pointer_sbrk(const PMEMoid &oid) noexcept
			: PMEMoid(oid)
		{}
		pointer_sbrk(const pointer_sbrk &) = default;
		pointer_sbrk &operator=(const pointer_sbrk &) = default;
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
