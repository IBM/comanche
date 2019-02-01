#ifndef COMANCHE_HSTORE_PMEM_H
#define COMANCHE_HSTORE_PMEM_H

#include <api/kvstore_itf.h> /* E_TOO_LARGE */

template <unsigned Offse>
	struct check_offset;
#if USE_CC_HEAP == 1
#elif USE_CC_HEAP == 2
template <>
    struct check_offset<0U>
    {
    };
#else /* USE_CC_HEAP */
template <>
    struct check_offset<48U>
    {
    };
#endif /* USE_CC_HEAP */

#include "allocator_pobj_cache_aligned.h"
#include "allocator_co.h"
#include "heap_co.h"
#include "palloc.h"
#include "persister_pmem.h"
#include "store_root.h"

#include "hstore_common.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#if defined __clang__
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#endif
#include <libpmemobj.h>
#include <libpmempool.h>
#include <libpmemobj/base.h>
#include <libpmem.h> /* pmem_persist */
#pragma GCC diagnostic pop
#include <boost/filesystem.hpp>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring> /* strerror, memcpy */
#include <memory> /* unique_ptr */
#include <new>
#include <map> /* session set */
#include <mutex> /* thread safe use of libpmempool/obj */

#define REGION_NAME "hstore-data"

using open_pool_handle = std::unique_ptr<PMEMobjpool, void(*)(PMEMobjpool *)>;

#include "hstore_open_pool.h"

using Persister = persister_pmem;

namespace
{
  TOID_DECLARE_ROOT(struct store_root_t);

  /* Some pmemobj calls are not thread-safe (PMEM issue 872).
   */
  std::mutex pmemobj_mutex;

  using pmemobj_guard_t = std::lock_guard<std::mutex>;

  PMEMobjpool *pmemobj_create_guarded(const char *path, const char *layout,
    size_t poolsize, mode_t mode)
  {
    pmemobj_guard_t g{pmemobj_mutex};
    return ::pmemobj_create(path, layout, poolsize, mode);
  }
  PMEMobjpool *pmemobj_open_guarded(const char *path, const char *layout)
  {
    pmemobj_guard_t g{pmemobj_mutex};
    return ::pmemobj_open(path, layout);
  }
  void pmemobj_close_guarded(PMEMobjpool *pop)
  {
    pmemobj_guard_t g{pmemobj_mutex};
    ::pmemobj_close(pop);
  }

  int check_pool(const char * path)
  {
    struct pmempool_check_args args;
    args.path = path;
    args.backup_path = NULL;
    args.pool_type = PMEMPOOL_POOL_TYPE_DETECT;
    args.flags =
      PMEMPOOL_CHECK_FORMAT_STR |
      PMEMPOOL_CHECK_REPAIR |
      PMEMPOOL_CHECK_VERBOSE;

    if (auto ppc = pmempool_check_init(&args, sizeof(args)))
    {
      /* perform check and repair, answer 'yes' for each question */
      while ( auto status = pmempool_check(ppc) ) {
        switch (status->type) {
        case PMEMPOOL_CHECK_MSG_TYPE_ERROR:
          printf("%s\n", status->str.msg);
          break;
        case PMEMPOOL_CHECK_MSG_TYPE_INFO:
          printf("%s\n", status->str.msg);
          break;
        case PMEMPOOL_CHECK_MSG_TYPE_QUESTION:
          printf("%s\n", status->str.msg);
          status->str.answer = "yes";
          break;
        default:
          pmempool_check_end(ppc);
          throw General_exception("pmempool_check failed %s", path);
        }
      }

      /* finalize the check and get the result */
      int ret = pmempool_check_end(ppc);
      switch (ret) {
      case PMEMPOOL_CHECK_RESULT_CONSISTENT:
      case PMEMPOOL_CHECK_RESULT_REPAIRED:
        return 0;
      }

      return 1;
    }

    PLOG("pmempool_check_init (%s) %s", path, strerror(errno));
    return -1;
  }

  store_root_t *read_root(TOID(struct store_root_t) &root)
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wpedantic"
    store_root_t *rt = D_RW(root);
#pragma GCC diagnostic pop
    return rt;
  }

  const store_root_t *read_const_root(TOID(struct store_root_t) &root)
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wpedantic"
    const store_root_t *rt = D_RO(root);
#pragma GCC diagnostic pop
    return rt;
  }

  auto Pmem_delete_and_recreate_pool(const char *path, const std::size_t size, const char *action) -> PMEMobjpool *
  {
    if ( 0 != pmempool_rm(path, PMEMPOOL_RM_FORCE | PMEMPOOL_RM_POOLSET_LOCAL))
      throw General_exception("pmempool_rm on (%s) failed: %x", path, pmemobj_errormsg());

    auto pop = pmemobj_create_guarded(path, REGION_NAME, size, 0666);
    if (not pop) {
      pop = pmemobj_create_guarded(path, REGION_NAME, 0, 0666); /* size = 0 for devdax */
      if (not pop)
        throw General_exception("failed to %s (%s) %s", action, path, pmemobj_errormsg());
    }
    return pop;
  }
}

auto Pmem_make_devdax_manager() -> std::shared_ptr<nupm::Devdax_manager>
{
  return std::shared_ptr<nupm::Devdax_manager>(nullptr);
}

namespace
{
  open_pool_handle create_or_open_pool(const std::string &dir_, const std::string &name_, std::size_t size_, bool option_DEBUG_)
  {
    std::string path = make_full_path(dir_, name_);
    open_pool_handle pop(nullptr, pmemobj_close_guarded);

    /* NOTE: conditions can change between the access call and the create/open call.
     * This code makes no provision for such a change.
     */
    if (::access(path.c_str(), F_OK) != 0) {
      if ( option_DEBUG_ )
        {
          PLOG(PREFIX "creating new pool: %s (%s) size=%lu"
               , __func__
               , name_.c_str()
               , path.c_str()
               , size_
               );
        }

      boost::filesystem::path p(path);
      boost::filesystem::create_directories(p.parent_path());

      pop.reset(pmemobj_create_guarded(path.c_str(), REGION_NAME, size_, 0666));
      if (not pop)
        {
          throw General_exception("failed to create new pool %s (%s)", path.c_str(), pmemobj_errormsg());
        }
    }
    else {
      if ( option_DEBUG_ )
        {
          PLOG(PREFIX "opening existing Pool: %s", __func__, path.c_str());
        }

      if (check_pool(path.c_str()) != 0)
        {
          pop.reset(Pmem_delete_and_recreate_pool(path.c_str(), size_, "create new pool"));
        }
      else {
        /* open existing */
        {
          pop.reset(pmemobj_open_guarded(path.c_str(), REGION_NAME));
        }
        if (not pop)
          {
            PWRN(PREFIX "erasing memory pool/partition: %s", __func__, path.c_str());
            /* try to delete pool and recreate */
            pop.reset(Pmem_delete_and_recreate_pool(path.c_str(), size_, "re-open or create new pool"));
          }
      }
    }
    return pop;
  }
}

auto Pmem_create_pool_check(const std::size_t size_) -> status_t
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  if (PMEMOBJ_MAX_ALLOC_SIZE < size_)
  {
    PWRN(
         PREFIX "object too large (max %zu, size %zu)"
         , __func__
         , PMEMOBJ_MAX_ALLOC_SIZE
         , size_
         );
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverflow"
    /* NOTE: E_TOO_LARGE may be negative, but pool_t is uint64_t */
    return uint64_t(Component::IKVStore::E_TOO_LARGE);
#pragma GCC diagnostic pop
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  if (size_ < PMEMOBJ_MIN_POOL) {
#pragma GCC diagnostic pop
    PWRN(PREFIX "object too small", __func__);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverflow"
    /* NOTE: E_BAD_PARAM may be negative, but pool_t is uint64_t */
    return uint64_t(Component::IKVStore::E_BAD_PARAM);
#pragma GCC diagnostic pop
  }
  return 0;
}

void Pmem_close_pool_check_pool(const std::string &path)
{
  if ( path != "" ) {
    if ( check_pool(path.c_str()) != 0 )
    {
      PLOG("pool check failed (%s) %s", path.c_str(), pmemobj_errormsg());
    }
  }
}

void Pmem_delete_pool(const std::string &path)
{
  constexpr int flags = 0;
  if ( 0 != pmempool_rm(path.c_str(), flags) ) {
    auto e = errno;
    throw
      General_exception(
                        "unable to delete pool (%s): pmem err %s errno %d (%s)"
                        , path.c_str()
                        , pmemobj_errormsg()
                        , e
                        , strerror(e)
                        );
  }
}

status_t Pmem_get_pool_regions(PMEMobjpool *pool, std::vector<::iovec>& out_regions)
{
  /* calls pmemobj extensions in modified version of PMDK */
  unsigned idx = 0;
  void * base = nullptr;
  size_t len = 0;

  while ( pmemobj_ex_pool_get_region(pool, idx, &base, &len) == 0 ) {
    assert(base);
    assert(len);
    out_regions.push_back(::iovec{base,len});
    base = nullptr;
    len = 0;
    idx++;
  }

  return S_OK;
}

#endif
