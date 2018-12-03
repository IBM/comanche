#include "palloc.h"

#include "trace_flags.h"

#include "pobj_bad_alloc.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#include <libpmemobj.h> /* pmemobj_{alloc,direct,free} */
#pragma GCC diagnostic pop
#if TRACE_PALLOC
#include <iostream> /* cerr */
#endif

PMEMoid palloc_inner(
	PMEMobjpool *pop_
	, std::size_t size_
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
	if ( 0 != pmemobj_alloc(pop_, &oid, size_, type_num_, ctor_, ctor_arg_) )
	{
		throw pobj_bad_alloc(errno);
	}
#if TRACE_PALLOC
	{
		void *ptr = pmemobj_direct(oid);
		std::cerr << __func__ << " " << use_ << " [" << ptr << ".."
			<< static_cast<void *>(static_cast<char *>(ptr)+size_) << ")\n";
	}
#endif
	return oid;
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
		std::cerr << __func__ << " " << why << " [" << ptr << "..)\n";
	}
#endif
	pmemobj_free(&oid);
}
