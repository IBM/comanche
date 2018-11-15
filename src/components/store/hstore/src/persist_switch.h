#ifndef _DANW_PERSIST_SWITCH_H_
#define _DANW_PERSIST_SWITCH_H_

#include <cstddef> /* ptrdiff_t */

template <typename A, bool enable>
	struct persist_switch;

/*
 * How to persist if allocator A has no persist method
 * (do nothing)
 */
template <typename A>
	struct persist_switch<A, false>
	{
		static void persist(const A &, const void *, const void *, const char *) noexcept
		{
		}
	};

/* How to persist if allocator A has a persist method */
template <typename A>
	struct persist_switch<A, true>
	{
		static void persist(const A &a, const void *first, const void *last, const char *what) noexcept
		{
			std::ptrdiff_t len = static_cast<const char *>(last) - static_cast<const char *>(first);
			if ( 0 < len )
			{
				a.persist(first, len, what);
			}
		}
	};

#endif
