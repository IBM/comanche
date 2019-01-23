#ifndef _DAWN_HSTORE_PALLOC_H_
#define _DAWN_HSTORE_PALLOC_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#include <libpmemobj.h> /* PMEMobjpool, PMEMoid, pmemobj_constr */
#pragma GCC diagnostic pop

#include <cstddef> /* size_t */
#include <tuple>

std::tuple<PMEMoid, std::size_t> palloc_inner(
	PMEMobjpool *pop
	, std::size_t size_min
	, std::size_t size_max
	, uint64_t type_num
	, pmemobj_constr ctor
	, void *ctor_arg
	, const char *use
);

template <typename T>
	PMEMoid palloc(
		PMEMobjpool *pop_
		, std::size_t size_
		, uint64_t type_num_
		, pmemobj_constr ctor_
		, const T &ctor_arg_
		, const char *use_
)
{
	return
		std::get<0>(
			palloc_inner(
				pop_
				, size_
				, size_
				, type_num_
				, ctor_
				, &const_cast<T &>(ctor_arg_)
				, use_
			)
		);
}

static inline PMEMoid palloc(
	PMEMobjpool *pop_
	, std::size_t size_
	, uint64_t type_num_
	, const char *use_
)
{
	return
		std::get<0>(
			palloc_inner(
				pop_
				, size_
				, size_
				, type_num_
				, nullptr
				, nullptr
				, use_
			)
		);
}

static inline std::tuple<PMEMoid, std::size_t> palloc(
	PMEMobjpool *pop_
	, std::size_t size_min_
	, std::size_t size_mac_
	, uint64_t type_num_
	, const char *use_
)
{
	return
		palloc_inner(
			pop_
			, size_min_
			, size_mac_
			, type_num_
			, nullptr
			, nullptr
			, use_
		);
}

void zfree(
	PMEMoid oid
	, const char *why
);

#endif
