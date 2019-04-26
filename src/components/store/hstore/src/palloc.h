/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


#ifndef _COMANCHE_HSTORE_PALLOC_H_
#define _COMANCHE_HSTORE_PALLOC_H_

#include "trace_flags.h"

#include "pobj_bad_alloc.h"

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
	PMEMobjpool *pop_
	, std::size_t size_min_
	, std::size_t size_max_
	, uint64_t type_num_
	, pmemobj_constr ctor_
	, void *ctor_arg_
	, const char *
#if TRACE_PALLOC
		use_
#endif
)
{
	PMEMoid oid;
	if ( size_max_ < size_min_ )
	{
		throw pobj_bad_alloc(0, 1, size_max_, size_min_, EINVAL);
	}
	while ( 0 != pmemobj_alloc(pop_, &oid, size_max_, type_num_, ctor_, ctor_arg_) )
	{
		size_max_ = size_max_ / 64U * 63U;
		if ( size_max_ < size_min_ )
		{
			throw pobj_bad_alloc(0, 1, size_max_, size_min_, errno);
		}
	}
#if TRACE_PALLOC
	{
		void *ptr = pmemobj_direct(oid);
		hop_hash_log::write(__func__, " " << use_, " [", ptr, ".."
			, static_cast<void *>(static_cast<char *>(ptr)+size_max_), ")"i
		);
	}
#endif
	return std::tuple<PMEMoid, std::size_t>(oid, size_max_);
}

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
	, const char *
#if TRACE_PALLOC
		why
#endif
)
{
#if TRACE_PALLOC
	{
		const auto ptr = pmemobj_direct(oid);
		hop_hash_log::write(__func__, " ", why, " [" <<,r << "..)");
	}
#endif
	pmemobj_free(&oid);
}

#endif
