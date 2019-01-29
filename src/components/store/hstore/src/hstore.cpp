#include "hstore.h"

#include "atomic_controller.h"
#include "hop_hash.h"
#include "palloc.h"
#include "perishable.h"
#include "persist_fixed_string.h"
#include "persister_pmem.h"
#include "allocator_pobj_cache_aligned.h"

#include <stdexcept>
#include <city.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/utils.h>
#include <core/cc_heap.h>

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
#include <memory> /* unique_ptr */
#include <new>
#include <map> /* session set */
#include <mutex> /* thread safe use of libpmempool/obj */

#define PREFIX "HSTORE : %s: "

#define REGION_NAME "hstore-data"

#define USE_CC_HEAP 0

using IKVStore = Component::IKVStore;

#if 0
/* thread-safe hash */
#include <mutex>
using hstore_shared_mutex = std::shared_timed_mutex;
static constexpr auto thread_model = IKVStore::THREAD_MODEL_MULTI_PER_POOL;
#else
/* not a thread-safe hash */
#include "dummy_shared_mutex.h"
using hstore_shared_mutex = dummy::shared_mutex;
static constexpr auto thread_model = IKVStore::THREAD_MODEL_SINGLE_PER_POOL;
#endif

using open_pool_handle = std::unique_ptr<PMEMobjpool, void(*)(PMEMobjpool *)>;

template<typename T>
  struct type_number;

template<> struct type_number<char> { static constexpr uint64_t value = 2; };

namespace
{
  constexpr bool option_DEBUG = false;
  namespace type_num
  {
    constexpr uint64_t persist = 1U;
  }
}

#if USE_CC_HEAP
using ALLOC_T = Core::CC_allocator<char, persister_pmem>;
using KEY_T = persist_fixed_string<char, ALLOC_T>;
using MAPPED_T = persist_fixed_string<char, ALLOC_T>;
#else /* USE_CC_HEAP */
using ALLOC_T = allocator_pobj_cache_aligned<char>;
using DEALLOC_T = typename ALLOC_T::deallocator_type;
using KEY_T = persist_fixed_string<char, DEALLOC_T>;
using MAPPED_T = persist_fixed_string<char, DEALLOC_T>;
#endif /* USE_CC_HEAP */

namespace
{
  struct pstr_hash
  {
    using argument_type = KEY_T;
    using result_type = std::uint64_t;
    result_type hf(const argument_type &s) const
    {
      return CityHash64(s.data(), s.size());
    }
  };

  using HASHER_T = pstr_hash;
}

#if USE_CC_HEAP
using allocator_segment_t =
  Core::CC_allocator<std::pair<const KEY_T, MAPPED_T>, persister_pmem>;

using allocator_atomic_t =
  Core::CC_allocator<impl::mod_control, persister_pmem>;
#else
using allocator_segment_t =
  allocator_pobj_cache_aligned<std::pair<const KEY_T, MAPPED_T>>;

using allocator_atomic_t =
  allocator_pobj_cache_aligned <impl::mod_control>;
template<> struct type_number<impl::mod_control> { static constexpr std::uint64_t value = 4; };
#endif /* USE_CC_HEAP */

using table_t =
  table<
  KEY_T
  , MAPPED_T
  , HASHER_T
  , std::equal_to<KEY_T>
  , allocator_segment_t
  , hstore_shared_mutex
  >;

template<> struct type_number<table_t::value_type> { static constexpr std::uint64_t value = 5; };
template<> struct type_number<table_t::base::persist_data_t::bucket_aligned_t> { static constexpr uint64_t value = 6; };

using persist_data_t = typename impl::persist_data<allocator_segment_t, table_t::value_type>;

namespace
{
  struct pc_al
  {
    persist_data_t _pc;
#if USE_CC_HEAP
    /* If using CC_allocator, area following _pc is a Core::cc_sbrk::state, which can be used to construct a CC_allocator */
    ALLOC_T get_alloc() { return ALLOC_T(static_cast<void *>(&_pc + 1)); }
#endif
  };

  struct store_root_t
  {
    /* A pointer so that null value can indicate no allocation.
     * Locates a pc_al_t.
     * - all allocated space can be accessed through pc
     * If using a CC_allocator:
     *   - space controlled by the allocator immediately follows the pc.
     *   - all free space can be accessed through allocator
     */
    PMEMoid pc;
  };

  TOID_DECLARE_ROOT(struct store_root_t);

  std::string make_full_path(const std::string &prefix, const std::string &suffix)
  {
    return prefix + ( prefix[prefix.length()-1] != '/' ? "/" : "") + suffix;
  }
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

  pc_al *map_open(TOID(struct store_root_t) &root)
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wpedantic"
    auto rt = D_RW(root);
#pragma GCC diagnostic pop
    PLOG(PREFIX "persist root addr %p", __func__, static_cast<const void *>(rt));
    auto apc = pmemobj_direct(rt->pc);
    PLOG(PREFIX "persist pc_al addr %p", __func__, static_cast<const void *>(apc));
    return static_cast<pc_al *>(apc);
  }

  void map_create(
    PMEMobjpool *pop_
    , TOID(struct store_root_t) &root
    , std::size_t
#if USE_CC_HEAP
        size_
#endif /* USE_CC_HEAP */
    , std::size_t expected_obj_count
    , bool verbose
    )
  {
    if ( verbose )
    {
      PLOG(
           PREFIX "root is empty: new hash required object count %zu"
           , __func__
           , expected_obj_count
           );
    }
  auto oid_and_size =
    palloc(
           pop_
#if USE_CC_HEAP
           , sizeof(pc_al) + 64U /* least acceptable size */
           , size_ /* preferred size */
#else /* USE_CC_HEAP */
           , sizeof(persist_data_t)
           , sizeof(persist_data_t)
#endif /* USE_CC_HEAP */
           , type_num::persist
           , "persist"
           );

    auto oid = std::get<0>(oid_and_size);
    pc_al *p = static_cast<pc_al *>(pmemobj_direct(oid));
#if USE_CC_HEAP
    auto actual_size = std::get<1>(oid_and_size);
    PLOG(PREFIX "createed pc_al at addr %p preferred size %zu size %zu", __func__, static_cast<const void *>(p), size_, actual_size);
    /* arguments to cc_malloc are the start of the free space (which cc_sbrk uses
     * for the "state" structure) and the size of the free space
     */
    ALLOC_T al(&p->_pc + 1, actual_size - sizeof(pc_al));
    new (&p->_pc) persist_data_t(
      expected_obj_count
      , table_t::allocator_type(al)
    );
    ::pmem_persist(p, sizeof *p);
#else /* USE_CC_HEAP */
    new (&p->_pc) persist_data_t(expected_obj_count, table_t::allocator_type{pop_});
    table_t::allocator_type{pop_}
      .persist(p, sizeof *p, "persist_data");
#endif /* USE_CC_HEAP */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wpedantic"
    D_RW(root)->pc = oid;
#pragma GCC diagnostic pop
  }

  pc_al *map_create_if_null(
                         PMEMobjpool *pop_
                         , TOID(struct store_root_t) &root
                         , std::size_t size_
                         , std::size_t expected_obj_count
                         , bool verbose
                         )
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
    const bool initialized = ! OID_IS_NULL(D_RO(root)->pc);
#pragma GCC diagnostic pop
    if ( ! initialized )
    {
      map_create(pop_, root, size_, expected_obj_count, verbose
      );
    }
    return map_open(root);
  }

  PMEMobjpool *delete_and_recreate_pool(const char *fullpath, const std::size_t size, const char *action)
  {
    if ( 0 != pmempool_rm(fullpath, PMEMPOOL_RM_FORCE | PMEMPOOL_RM_POOLSET_LOCAL))
      throw General_exception("pmempool_rm on (%s) failed: %x", fullpath, pmemobj_errormsg());

    auto pop = pmemobj_create_guarded(fullpath, REGION_NAME, size, 0666);
    if (not pop) {
      pop = pmemobj_create_guarded(fullpath, REGION_NAME, 0, 0666); /* size = 0 for devdax */
      if (not pop)
        throw General_exception("failed to %s (%s) %s", action, fullpath, pmemobj_errormsg());
    }
    return pop;
  }

  struct tls_cache_t {
    session *recent_session;
  };
}

class session
{
  TOID(struct store_root_t) _root;
  std::string               _dir;
  std::string               _name;
  open_pool_handle          _pop;
  ALLOC_T                   _heap;
  table_t                   _map;
  impl::atomic_controller<table_t> _atomic_state;
public:
  explicit session(
                        TOID(struct store_root_t) &root_
                        , open_pool_handle &&pop_
                        , const std::string &dir_
                        , const std::string &name_
                        , pc_al *pc_al_
                        )
    : _root(root_)
    , _dir(dir_)
    , _name(name_)
    , _pop(std::move(pop_))
#if USE_CC_HEAP
    , _heap(pc_al_->get_alloc())
#else /* USE_CC_HEAP */
    , _heap(ALLOC_T(pmem_pool()))
#endif /* USE_CC_HEAP */
    , _map(&pc_al_->_pc, _heap)
    , _atomic_state(pc_al_->_pc, _map)
  {}

  session(const session &) = delete;
  session& operator=(const session &) = delete;
  auto allocator() const { return _heap; }
  PMEMobjpool *pmem_pool() const { return _pop.get(); }
#if 1
  /* delete_pool only */
  const std::string &dir() const noexcept { return _dir; }
  const std::string &name() const noexcept { return _name; }
#endif
  table_t &map() noexcept { return _map; }
  const table_t &map() const noexcept { return _map; }

  auto enter(
             KEY_T &key
             , std::vector<Component::IKVStore::Operation *>::const_iterator first
             , std::vector<Component::IKVStore::Operation *>::const_iterator last
             ) -> Component::status_t
  {
    return _atomic_state.enter(allocator(), key, first, last);
  }
};

/* globals */
thread_local tls_cache_t tls_cache = { nullptr };

auto hstore::locate_session(const IKVStore::pool_t pid) -> session &
{
  auto *const s = reinterpret_cast<struct session *>(pid);
  if ( s == nullptr || s != tls_cache.recent_session )
    {
      std::unique_lock<std::mutex> sessions_lk(sessions_mutex);
      auto ps = g_sessions.find(s);
      if ( ps == g_sessions.end() )
        {
          throw API_exception(PREFIX "invalid pool identifier %p", __func__, s);
        }
      tls_cache.recent_session = ps->second.get();
    }
  return *tls_cache.recent_session;
}

auto hstore::move_session(const IKVStore::pool_t pid) -> std::unique_ptr<session>
{
  auto *const s = reinterpret_cast<struct session *>(pid);

  std::unique_lock<std::mutex> sessions_lk(sessions_mutex);
  auto ps = g_sessions.find(s);
  if ( ps == g_sessions.end() )
    {
      throw API_exception(PREFIX "invalid pool identifier %p", __func__, s);
    }

  if ( s == tls_cache.recent_session ) { tls_cache.recent_session = nullptr; }
  auto s2 = std::move(ps->second);
  g_sessions.erase(ps);
  return s2;
}

hstore::hstore(const std::string & /* owner */, const std::string & /* name */)
  : sessions_mutex{}
  , g_sessions{}
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  PLOG("PMEMOBJ_MAX_ALLOC_SIZE: %lu MB", REDUCE_MB(PMEMOBJ_MAX_ALLOC_SIZE));
#pragma GCC diagnostic pop
}

hstore::~hstore()
{
}

auto hstore::thread_safety() const -> status_t
{
  return thread_model;
}

auto hstore::create_pool(
                         const std::string &path,
                         const std::string &name,
                         const std::size_t size_,
                         unsigned int /* flags */,
                         uint64_t expected_obj_count /* args */) -> pool_t
{
  if ( option_DEBUG )
    {
      PLOG(PREFIX "path=%s pool_name=%s", __func__, path.c_str(), name.c_str());
    }
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
      /* NOTE: E_TOO_LARGE may be negative, but pool_t is uint64_t */
      return uint64_t(E_TOO_LARGE);
    }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  if (size_ < PMEMOBJ_MIN_POOL) {
#pragma GCC diagnostic pop
    PWRN(PREFIX "object too small", __func__);
    /* NOTE: E_BAD_PARAM may be negative, but pool_t is uint64_t */
    return uint64_t(E_BAD_PARAM);
  }

  std::string fullpath = make_full_path(path, name);

  open_pool_handle pop(nullptr, pmemobj_close_guarded);

  /* NOTE: conditions can change between the access call and the create/open call.
   * This code makes no provision for such a change.
   */
  if (access(fullpath.c_str(), F_OK) != 0) {
    if ( option_DEBUG )
      {
        PLOG(PREFIX "creating new pool: %s (%s) size=%lu"
             , __func__
             , name.c_str()
             , fullpath.c_str()
             , size_
             );
      }

    boost::filesystem::path p(fullpath);
    boost::filesystem::create_directories(p.parent_path());

    pop.reset(pmemobj_create_guarded(fullpath.c_str(), REGION_NAME, size_, 0666));
    if (not pop)
      {
        throw General_exception("failed to create new pool %s (%s)", fullpath.c_str(), pmemobj_errormsg());
      }
  }
  else {
    if ( option_DEBUG )
      {
        PLOG(PREFIX "opening existing Pool: %s", __func__, fullpath.c_str());
      }

    if (check_pool(fullpath.c_str()) != 0)
      {
        pop.reset(delete_and_recreate_pool(fullpath.c_str(), size_, "create new pool"));
      }
    else {
      /* open existing */
      {
        pop.reset(pmemobj_open_guarded(fullpath.c_str(), REGION_NAME));
      }
      if (not pop)
        {
          PWRN(PREFIX "erasing memory pool/partition: %s", __func__, fullpath.c_str());
          /* try to delete pool and recreate */
          pop.reset(delete_and_recreate_pool(fullpath.c_str(), size_, "re-open or create new pool"));
        }
    }
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  TOID(struct store_root_t) root = POBJ_ROOT(pop.get(), struct store_root_t);
#pragma GCC diagnostic pop
  assert(!TOID_IS_NULL(root));

  auto pc =
    map_create_if_null(
      pop.get(), root, size_, expected_obj_count, option_DEBUG
    );
  auto s = std::make_unique<session>(root, std::move(pop), path, name, pc);
  auto p = s.get();
  std::unique_lock<std::mutex> sessions_lk(sessions_mutex);
  g_sessions.emplace(p, std::move(s));

  return reinterpret_cast<IKVStore::pool_t>(p);
}

bool is_header_compact(PMEMobjpool *pop, unsigned class_id)
{
  struct pobj_alloc_class_desc desc;
  auto r =
    pmemobj_ctl_get(
                    pop
                    , ("heap.alloc_class." + std::to_string(class_id) + ".desc").c_str()
                    , &desc
                    );
  if ( r != 0 )
    {
      throw General_exception("class header test failed");
    }
#if 0
  std::cerr << "class:"
            << " unit size " << desc.unit_size
            << " alignment " << desc.alignment
            << " units_per_block " << desc.units_per_block
            << " header_type " << desc.header_type
            << " class_id " << desc.class_id
            << "\n";
#endif
  return desc.header_type == POBJ_HEADER_COMPACT;
}

auto hstore::open_pool(const std::string &path,
                       const std::string &name,
                       unsigned int /* flags */) -> pool_t
{
  if (access(path.c_str(), F_OK) != 0)
    {
      throw API_exception("Pool %s:%s does not exist", path.c_str(), name.c_str());
    }

  std::string fullpath = make_full_path(path, name);

  /* check integrity first */
  if (check_pool(fullpath.c_str()) != 0)
    {
      throw General_exception("pool check failed");
    }

  if (
      auto pop =
        open_pool_handle(pmemobj_open_guarded(fullpath.c_str(), REGION_NAME), pmemobj_close_guarded)
      )
    {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
      TOID(struct store_root_t) root = POBJ_ROOT(pop.get(), struct store_root_t);
#pragma GCC diagnostic pop
      if (TOID_IS_NULL(root))
        {
          throw General_exception("Root is NULL!");
        }

      auto pc = map_open(root);
      if ( ! pc )
      {
        throw General_exception("failed to re-open pool (not initialized)");
      }

      auto s = std::make_unique<session>(root, std::move(pop), path, name, pc);
      auto p = s.get();
      std::unique_lock<std::mutex> sessions_lk(sessions_mutex);
      g_sessions.emplace(p, std::move(s));

      return reinterpret_cast<IKVStore::pool_t>(p);
    }
  throw General_exception("failed to re-open pool - %s", pmemobj_errormsg());
}

void hstore::close_pool(const pool_t pid)
{
  std::string path;
  try
    {

      auto session = move_session(pid);
      if ( option_DEBUG )
        {
          PLOG(PREFIX "closed pool (%lx)", __func__, pid);
        }
    }
  catch ( const API_exception &e )
    {
      throw API_exception("%s in %s", e.cause(), __func__);
    }
  if ( path != "" ) {
    if ( check_pool(path.c_str()) != 0 )
    {
      PLOG("pool check failed (%s) %s", path.c_str(), pmemobj_errormsg());
    }
  }
}

void hstore::delete_pool(const std::string &dir, const std::string &name)
{
  const int flags = 0;
  auto path = make_full_path(dir, name);
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

  if ( option_DEBUG )
    {
      PLOG("pool deleted: %s/%s", dir.c_str(), name.c_str());
    }
}

void hstore::delete_pool(const pool_t pid)
  try
    {
      /* Not sure why a session would have to be open in order to erase a pool,
       * but the kvstore interface requires it.
       */
      std::string dir;
      std::string name;
      {
        auto session = move_session(pid);
        dir = session->dir();
        name = session->name();
      }
      delete_pool(dir, name);
    }
  catch ( const API_exception &e )
    {
      throw API_exception("%s in %s", e.cause(), __func__);
    }

auto hstore::put(const pool_t pool,
                 const std::string &key,
                 const void * value,
                 const std::size_t value_len) -> status_t
{
  if ( option_DEBUG ) {
    PLOG(
         PREFIX "(key=%s) (value=%.*s)"
         , __func__
         , key.c_str()
         , int(value_len)
         , static_cast<const char*>(value)
         );
    assert(0 < value_len);
  }

  if(value == nullptr)
    throw std::invalid_argument("value argument is null");

  auto &session = locate_session(pool);

  auto cvalue = static_cast<const char *>(value);

  const auto i =
    session.map().emplace(
                          std::piecewise_construct
                          , std::forward_as_tuple(key.begin(), key.end(), session.allocator())
                          , std::forward_as_tuple(cvalue, cvalue + value_len, session.allocator())
                          );
  return i.second ? S_OK : update_by_issue_41(pool, key, value, value_len,  i.first->second.data(), i.first->second.size());
}

auto hstore::update_by_issue_41(const pool_t pool,
                 const std::string &key,
                 const void * value,
                 const std::size_t value_len,
                 void * old_value,
                 const std::size_t old_value_len
) -> status_t
{
  /* hstore issue 41: "a put should replace any existing k,v pairs that match. If the new put is a different size, then the object should be reallocated. If the new put is the same size, then it should be updated in place." */
  if ( value_len != old_value_len )
  {
    this->erase(pool, key);
    return put(pool, key, value, value_len);
  }
  else {
    std::memcpy(old_value, value, value_len);
    return S_OK;
#if 0
    std::vector<std::unique_ptr<IKVStore::Operation>> v;
    v.emplace_back(std::make_unique<IKVStore::Operation_write>(0, value_len, value));
    std::vector<IKVStore::Operation *> v2;
    std::transform(v.begin(), v.end(), std::back_inserter(v2), [] (const auto &i) { return i.get(); });
    return this->atomic_update(
      pool
      , key
      , v2
      , false
    );
#endif
  }
}


status_t hstore::get_pool_regions(const pool_t pool, std::vector<::iovec>& out_regions)
{
  auto &session = locate_session(pool);
  const auto& pop = session.pmem_pool();

  /* calls pmemobj extensions in modified version of PMDK */
  unsigned idx = 0;
  void * base = nullptr;
  size_t len = 0;

  while ( pmemobj_ex_pool_get_region(pop, idx, &base, &len) == 0 ) {
    assert(base);
    assert(len);
    out_regions.push_back(::iovec{base,len});
    base = nullptr;
    len = 0;
    idx++;
  }

  return S_OK;
}

auto hstore::put_direct(const pool_t pool,
                        const std::string& key,
                        const void * value,
                        const std::size_t value_len,
                        memory_handle_t) -> status_t
{
  return put(pool, key, value, value_len);
}

auto hstore::get(const pool_t pool,
                 const std::string &key,
                 void*& out_value,
                 std::size_t& out_value_len) -> status_t
{
#if 0
  PLOG(PREFIX " get(%s)", __func__, key.c_str());
#endif
  try
    {
      const auto &session = locate_session(pool);
      auto p_key = KEY_T(key.begin(), key.end(), session.allocator());
      auto &v = session.map().at(p_key);
      out_value_len = v.size();
      out_value = malloc(out_value_len);
      if ( ! out_value )
        {
          throw std::bad_alloc();
        }
      memcpy(out_value, v.data(), out_value_len);
      return S_OK;
    }
  catch ( std::out_of_range & )
    {
      return E_KEY_NOT_FOUND;
    }
  catch (...) {
    throw General_exception(PREFIX "failed unexpectedly", __func__);
  }
}

auto hstore::get_direct(const pool_t pool,
                        const std::string & key,
                        void* out_value,
                        std::size_t& out_value_len,
                        Component::IKVStore::memory_handle_t) -> status_t
  try {
    const auto &session = locate_session(pool);
    auto p_key = KEY_T(key.begin(), key.end(), session.allocator());

    auto &v = session.map().at(p_key);

    auto value_len = v.size();
    if (out_value_len < value_len)
      {
        /* NOTE: it might be helpful to tell the caller how large a buffer we need,
         * but that dones not seem to be expected.
         */
        PWRN(PREFIX "failed; insufficient buffer", __func__);
        return E_INSUFFICIENT_BUFFER;
      }

    out_value_len = value_len;

    assert(out_value);

    /* memcpy for moment
     */
    memcpy(out_value, v.data(), out_value_len);
    if ( option_DEBUG )
      {
        PLOG(
             PREFIX "value_len=%lu value=(%s)", __func__
             , v.size()
             , static_cast<char*>(out_value)
             );
      }
    return S_OK;
  }
  catch ( const std::out_of_range & )
    {
      return E_KEY_NOT_FOUND;
    }
  catch(...) {
    throw General_exception("get_direct failed unexpectedly");
  }

#if 0
namespace
{
class lock
  : public IKVStore::Opaque_key
  , public std::string
{
public:
  lock(const std::string &)
    : std::string(s)
    , IKVStore::Opaque_key{}
  {}
};
}
#endif

namespace
{
bool try_lock(table_t &map, hstore::lock_type_t type, const KEY_T &p_key)
{
  return
    type == IKVStore::STORE_LOCK_READ
    ? map.lock_shared(p_key)
    : map.lock_unique(p_key)
    ;
}
}

auto hstore::lock(const pool_t pool,
                  const std::string &key,
                  lock_type_t type,
                  void *& out_value,
                  std::size_t & out_value_len) -> key_t
{
  auto &session = locate_session(pool);
  const auto p_key = KEY_T(key.begin(), key.end(), session.allocator());

  try
    {
      MAPPED_T &val = session.map().at(p_key);
      if ( ! try_lock(session.map(), type, p_key) )
        {
          return KEY_NONE;
        }
      out_value = val.data();
      out_value_len = val.size();
    }
  catch ( std::out_of_range & )
    {
      /* if the key is not found, we create it and
         allocate value space equal in size to out_value_len
      */

      if ( option_DEBUG )
        {
          PLOG(PREFIX "allocating object %lu bytes", __func__, out_value_len);
        }

      auto r =
        session.map().emplace(
                              std::piecewise_construct
                              , std::forward_as_tuple(p_key)
                              , std::forward_as_tuple(out_value_len, session.allocator())
                              );

      if ( ! r.second )
        {
          return KEY_NONE;
        }

      out_value = r.first->second.data();
      out_value_len = r.first->second.size();
    }
  return reinterpret_cast<key_t>(new std::string(key));
}


auto hstore::unlock(const pool_t pool,
                    key_t key_) -> status_t
{
  std::string *key = reinterpret_cast<std::string *>(key_);

  if ( key )
    {
      try {
        auto &session = locate_session(pool);
        auto p_key = KEY_T(key->begin(), key->end(), session.allocator());

        session.map().unlock(p_key);
      }
      catch ( const std::out_of_range &e )
        {
          return E_KEY_NOT_FOUND;
        }
      catch(...) {
        throw General_exception(PREFIX "failed unexpectedly", __func__);
      }
      delete key;
    }

  return S_OK;
}

class maybe_lock
{
  table_t &_map;
  const KEY_T &_key;
  bool _taken;
public:
  maybe_lock(table_t &map_, const KEY_T &pkey_, bool take_)
    : _map(map_)
    , _key(pkey_)
    , _taken(false)
  {
    if ( take_ )
      {
        if( ! _map.lock_unique(_key) )
          {
            throw General_exception("unable to get write lock");
          }
        _taken = true;
      }
  }
  ~maybe_lock()
  {
    if ( _taken )
      {
        _map.unlock(_key); /* release lock */
      }
  }
};

auto hstore::apply(
                   const pool_t pool,
                   const std::string &key,
                   std::function<void(void*,std::size_t)> functor,
                   std::size_t object_size,
                   bool take_lock
                   ) -> status_t
{
  auto &session = locate_session(pool);
  MAPPED_T *val;
  auto p_key = KEY_T(key.begin(), key.end(), session.allocator());
  try
    {
      val = &session.map().at(p_key);
    }
  catch ( const std::out_of_range & )
    {
      /* if the key is not found, we create it and
         allocate value space equal in size to out_value_len
      */

      if ( option_DEBUG )
        {
          PLOG(PREFIX "allocating object %lu bytes", __func__, object_size);
        }

      auto r =
        session.map().emplace(
                              std::piecewise_construct
                              , std::forward_as_tuple(p_key)
                              , std::forward_as_tuple(object_size, session.allocator())
                              );
      if ( ! r.second )
        {
          return E_KEY_NOT_FOUND;
        }
      val = &(*r.first).second;
    }

  auto data = static_cast<char *>(val->data());
  auto data_len = val->size();

  maybe_lock m(session.map(), p_key, take_lock);

  functor(data, data_len);

  return S_OK;
}

auto hstore::erase(const pool_t pool,
                   const std::string &key
                   ) -> status_t
{
  try {
    auto &session = locate_session(pool);
    auto p_key = KEY_T(key.begin(), key.end(), session.allocator());
    return
      session.map().erase(p_key) == 0
      ? E_KEY_NOT_FOUND
      : S_OK
      ;
  }
  catch(...) {
    throw General_exception("hm_XXX_remove failed unexpectedly");
  }
}

std::size_t hstore::count(const pool_t pool)
{
  const auto &session = locate_session(pool);
  return session.map().size();
}

void hstore::debug(const pool_t pool, const unsigned cmd, const uint64_t arg)
{
  switch ( cmd )
    {
    case 0:
      perishable::enable(bool(arg));
      break;
    case 1:
      perishable::reset(arg);
      break;
    case 2:
      {
        const auto &session = locate_session(pool);
        table_t::size_type count = 0;
        /* bucket counter */
        for (
             auto n = session.map().bucket_count()
               ; n != 0
               ; --n
             )
          {
            auto last = session.map().end(n-1);
            for ( auto first = session.map().begin(n-1); first != last; ++first )
              {
                ++count;
              }
          }
        *reinterpret_cast<table_t::size_type *>(arg) = count;
      }
      break;
    default:
      break;
    };
#if 0
  auto &session = locate_session(pool);

  auto& root = session.root;
  auto& pop = session.pmem_pool();

  HM_CMD(pop, D_RO(root)->map(), cmd, arg);
#endif
}

namespace
{
/* Return value not set. Ignored?? */
int _functor(
             const std::string &key
             , MAPPED_T &m
             , std::function
             <
             int(const std::string &key, const void *val, std::size_t val_len)
             > *lambda)
{
  assert(lambda);
  (*lambda)(key, m.data(), m.size());
  return 0;
}
}

auto hstore::map(
                 pool_t pool,
                 std::function
                 <
                 int(const std::string &key, const void *val, std::size_t val_len)
                 > function
                 ) -> status_t
{
  auto &session = locate_session(pool);

  for ( auto &mt : session.map() )
    {
      const auto &pstring = mt.first;
      std::string s(static_cast<const char *>(pstring.data()), pstring.size());
      _functor(s, mt.second, &function);
    }

  return S_OK;
}

auto hstore::atomic_update(
                           const pool_t pool
                           , const std::string& key
                           , const std::vector<Operation *> &op_vector
                           , const bool take_lock) -> status_t
  try
    {
      auto &session = locate_session(pool);

      auto p_key = KEY_T(key.begin(), key.end(), session.allocator());

      maybe_lock m(session.map(), p_key, take_lock);

      return session.enter(p_key, op_vector.begin(), op_vector.end());
    }
  catch ( std::exception & )
    {
      return E_FAIL;
    }
