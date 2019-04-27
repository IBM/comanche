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


#ifndef COMANCHE_HSTORE_PMEM_H
#define COMANCHE_HSTORE_PMEM_H

#include "hstore_pool_manager.h"

#include <api/kvstore_itf.h> /* E_TOO_LARGE, E_BAD_PARAM */

#include "allocator_co.h"
#include "heap_co.h"
#include "palloc.h"
#include "persister_pmem.h"
#include "store_root.h"

#include "hstore_common.h"

#include "hstore_open_pool.h"
#include "hstore_session.h"

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

using Persister = persister_pmem;

namespace
{
  const char *REGION_NAME = "hstore-data";

  struct root_anchors
  {
    persist_data_t *persist_data_ptr;
    void *heap_ptr;
  };

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

  auto delete_and_recreate_pool(const char *path, const std::size_t size, const char *action) -> PMEMobjpool *
  {
    if ( 0 != pmempool_rm(path, PMEMPOOL_RM_FORCE | PMEMPOOL_RM_POOLSET_LOCAL))
      throw General_exception("pmempool_rm on (%s) failed: %" PRIxIKVSTORE_POOL_T, path, pmemobj_errormsg());

    auto pop = pmemobj_create_guarded(path, REGION_NAME, size, 0666);
    if (not pop) {
      pop = pmemobj_create_guarded(path, REGION_NAME, 0, 0666); /* size = 0 for devdax */
      if (not pop)
        throw General_exception("failed to %s (%s) %s", action, path, pmemobj_errormsg());
    }
    return pop;
  }
}

class hstore_pmem
  : public pool_manager
{
public:
  using open_pool_handle = std::unique_ptr<PMEMobjpool, void(*)(PMEMobjpool *)>;
private:

  auto map_open(TOID(struct store_root_t) &root) -> root_anchors
  {
    auto rt = read_const_root(root);
    PLOG(PREFIX "persist root addr %p", __func__, static_cast<const void *>(rt));
    auto apc = pmemobj_direct(rt->persist_oid);
    auto heap = pmemobj_direct(rt->heap_oid);
    PLOG(PREFIX "persist data addr %p", __func__, static_cast<const void *>(apc));
    PLOG(PREFIX "persist heap addr %p", __func__, static_cast<const void *>(heap));
    return root_anchors{static_cast<persist_data_t *>(apc), static_cast<void *>(heap)};
  }

  void map_create(
    PMEMobjpool *pop_
    , TOID(struct store_root_t) &root
    , std::size_t
#if USE_CC_HEAP == 1 || USE_CC_HEAP == 2
        size_
#endif /* USE_CC_HEAP */
    , std::size_t expected_obj_count
    )
  {
    if ( debug() )
    {
      PLOG(
           PREFIX "root is empty: new hash required object count %zu"
           , __func__
           , expected_obj_count
           );
    }
  auto persist_oid =
    palloc(
           pop_
           , sizeof(persist_data_t)
           , type_num::persist
           , "persist"
           );
    auto *p = static_cast<persist_data_t *>(pmemobj_direct(persist_oid));
    PLOG(PREFIX "created persist_data ptr at addr %p", __func__, static_cast<const void *>(p));

#if USE_CC_HEAP == 1
  auto heap_oid_and_size =
    palloc(
           pop_
           , 64U /* least acceptable size */
           , size_ /* preferred size */
           , type_num::heap
           , "heap"
           );

    auto heap_oid = std::get<0>(heap_oid_and_size);
    auto *a = static_cast<void *>(pmemobj_direct(heap_oid));
    auto actual_size = std::get<1>(heap_oid_and_size);
    PLOG(PREFIX "created heap at addr %p preferred size 0x%zx actual size 0x%zx", __func__, static_cast<const void *>(a), size_, actual_size);
    /* arguments to cc_malloc are the start of the free space (which cc_sbrk uses
     * for the "state" structure) and the size of the free space
     */
    auto al = new (a) Core::cc_alloc(static_cast<char *>(a) + sizeof(Core::cc_alloc), actual_size - sizeof(Core::cc_alloc));
    new (p) persist_data_t(
      expected_obj_count
      , table_t::allocator_type(*al)
    );
    Persister::persist(p, sizeof *p);
#elif USE_CC_HEAP == 2
    auto heap_oid_and_size =
      palloc(
             pop_
             , 64U /* least acceptable size */
             , size_ /* preferred size */
             , type_num::heap
             , "heap"
             );

    auto heap_oid = std::get<0>(heap_oid_and_size);
    auto *a = static_cast<void *>(pmemobj_direct(heap_oid));
    auto actual_size = std::get<1>(heap_oid_and_size);
    PLOG(PREFIX "createed heap at addr %p preferred size %zu actual size %zu", __func__, static_cast<const void *>(a), size_, actual_size);
    /* arguments to cc_malloc are the start of the free space (which cc_sbrk uses
     * for the "state" structure) and the size of the free space
     */
    auto al = new (a) Core::heap_co(heap_oid, actual_size, sizeof(Core::heap_co));
    new (p) persist_data_t(
      expected_obj_count
      , table_t::allocator_type(*al)
    );
    Persister::persist(p, sizeof *p);
#else /* USE_CC_HEAP */
    new (p) persist_data_t(expected_obj_count, table_t::allocator_type{pop_});
    table_t::allocator_type{pop_}
      .persist(p, sizeof *p, "persist_data");
#endif /* USE_CC_HEAP */

#if USE_CC_HEAP == 1
    read_root(root)->heap_oid = heap_oid;
#elif USE_CC_HEAP == 2
    read_root(root)->heap_oid = heap_oid;
#else /* USE_CC_HEAP */
#endif /* USE_CC_HEAP */
    read_root(root)->persist_oid = persist_oid;
  }

  auto map_create_if_null(
                         PMEMobjpool *pop_
                         , TOID(struct store_root_t) &root
                         , std::size_t size_
                         , std::size_t expected_obj_count
                         ) -> root_anchors
  {
    const bool initialized = ! OID_IS_NULL(read_const_root(root)->persist_oid);
    if ( ! initialized )
    {
      map_create(pop_, root, size_, expected_obj_count);
    }
    return map_open(root);
  }
  open_pool_handle pool_create_or_open(const pool_path &path_, std::size_t size_)
  {
    open_pool_handle pop(nullptr, pmemobj_close_guarded);

    /* NOTE: conditions can change between the access call and the create/open call.
     * This code makes no provision for such a change.
     */
    if (::access(path_.str().c_str(), F_OK) != 0) {
      if ( debug() )
        {
          PLOG(PREFIX "creating new pool: %s (%s) size=%zu"
               , __func__
               , path_.name().c_str()
               , path_.str().c_str()
               , size_
               );
        }

      boost::filesystem::path p(path_.str());
      boost::filesystem::create_directories(p.parent_path());

      pop.reset(pmemobj_create_guarded(path_.str().c_str(), REGION_NAME, size_, 0666));
      if (not pop)
        {
          throw General_exception("failed to create new pool %s (%s)", path_.str().c_str(), pmemobj_errormsg());
        }
    }
    else {
      if ( debug() )
        {
          PLOG(PREFIX "opening existing Pool: %s", __func__, path_.str().c_str());
        }

      if (check_pool(path_.str().c_str()) != 0)
        {
          pop.reset(delete_and_recreate_pool(path_.str().c_str(), size_, "create new pool"));
        }
      else {
        /* open existing */
        {
          pop.reset(pmemobj_open_guarded(path_.str().c_str(), REGION_NAME));
        }
        if (not pop)
          {
            PWRN(PREFIX "erasing memory pool/partition: %s", __func__, path_.str().c_str());
            /* try to delete pool and recreate */
            pop.reset(delete_and_recreate_pool(path_.str().c_str(), size_, "re-open or create new pool"));
          }
      }
    }
    return pop;
  }
public:

  hstore_pmem(const std::string &, const std::string &, bool debug_)
    : pool_manager(debug_)
  {}

  auto pool_create_check(const std::size_t size_) -> status_t override
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
    return S_OK;
  }

  auto pool_create(const pool_path &path_, std::size_t size_, std::size_t expected_obj_count) -> std::unique_ptr<tracked_pool> override
  {
    open_pool_handle pop = pool_create_or_open(path_, size_);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
    TOID(struct store_root_t) root = POBJ_ROOT(pop.get(), struct store_root_t);
#pragma GCC diagnostic pop
    assert(!TOID_IS_NULL(root));

    auto pc =
      map_create_if_null(
        pop.get(), root, size_, expected_obj_count
      );
    auto heap_oid = read_const_root(root)->heap_oid;
    return std::make_unique<session<open_pool_handle, ALLOC_T, table_t>>(heap_oid, path_, std::move(pop), pc.persist_data_ptr);
  }

  auto pool_open(const pool_path &path_) -> std::unique_ptr<tracked_pool> override
  {
    if (access(path_.str().c_str(), F_OK) != 0)
    {
      throw API_exception("Pool %s does not exist", path_.stri().c_str());
    }

    /* check integrity first */
    if (check_pool(path_.str().c_str()) != 0)
    {
      throw General_exception("pool check failed");
    }

    auto pop = open_pool_handle(pmemobj_open_guarded(path_.str().c_str(), REGION_NAME), pmemobj_close_guarded);
    if ( ! pop )
    {
      throw General_exception("failed to re-open pool %s - %s", path_.str().c_str(), pmemobj_errormsg());
    }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
    TOID(struct store_root_t) root = POBJ_ROOT(pop.get(), struct store_root_t);
#pragma GCC diagnostic pop
    if (TOID_IS_NULL(root))
    {
      throw General_exception("Root is NULL!");
    }

    auto anchors = map_open(root);

    if ( ! anchors.persist_data_ptr )
    {
      throw General_exception("failed to re-open pool (not initialized)");
    }

    /* open_pool returns a ::session.
     * If the session constructor throws an exception opening an pool you wish to delete,
     * use the form of delete_pool which does not require an open pool.
     */
    auto heap_oid = read_const_root(root)->heap_oid;
    return std::make_unique<session<open_pool_handle, ALLOC_T, table_t>>(heap_oid, path_, std::move(pop), anchors.persist_data_ptr);
  }

  void pool_close_check(const std::string &path)
  {
    if ( path != "" ) {
      if ( check_pool(path.c_str()) != 0 )
      {
        PLOG("pool check failed (%s) %s", path.c_str(), pmemobj_errormsg());
      }
    }
  }

  void pool_delete(const pool_path &path) override
  {
    constexpr int flags = 0;
    if ( 0 != pmempool_rm(path.str().c_str(), flags) ) {
      auto e = errno;
      throw
        General_exception(
                          "unable to delete pool (%s): pmem err %s errno %d (%s)"
                          , path.str().c_str()
                          , pmemobj_errormsg()
                          , e
                          , strerror(e)
                          );
    }
  }

  std::vector<::iovec> pool_get_regions(void *pool_) override
  {
    PMEMobjpool *pool = static_cast<PMEMobjpool *>(pool_);
    /* calls pmemobj extensions in modified version of PMDK */
    void * base = nullptr;
    size_t len = 0;

    std::vector<::iovec> resions;
    for ( unsigned idx = 0; pmemobj_ex_pool_get_region(pool, idx, &base, &len) == 0, ++idx )
    {
      assert(base);
      assert(len);
      regions.push_back(::iovec{base,len});
    }

    return regions;
  }
};

#endif
