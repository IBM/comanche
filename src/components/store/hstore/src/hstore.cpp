#include "hstore.h"

#include "atomic_controller.h"
#include "hop_hash.h"
#include "palloc.h"
#include "perishable.h"
#include "persist_fixed_string.h"
#include "pobj_allocator.h"

#include <city.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/utils.h>

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
#include <cerrno>
#include <memory> /* unique_ptr */
#include <new>
#include <map> /* session set */

#define PREFIX "HSTORE : %s: "

#define REGION_NAME "hstore-default"

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

namespace
{
  namespace type_num
  {
    constexpr uint64_t persist = 1U;
    constexpr uint64_t table = 2U;
    constexpr uint64_t key = 3U;
    constexpr uint64_t mapped = 4U;
  }

  using KEY_T = persist_fixed_string<char>;

  struct pstr_hash
  {
    using argument_type = KEY_T;
    using result_type = std::uint64_t;
    result_type hf(argument_type s) const
    {
      return CityHash64(s.data(), s.size());
    }
  };

  using HASHER_T = pstr_hash;

  using allocator_segment_t =
    pobj_cache_aligned_allocator
    <
      std::pair<const KEY_T, persist_fixed_string<char>>
    >;

  using allocator_atomic_t =
    pobj_cache_aligned_allocator<impl::mod_control>;

  using table_t =
    table<
      KEY_T
      , persist_fixed_string<char>
      , HASHER_T
      , std::equal_to<KEY_T>
      , allocator_segment_t
      , hstore_shared_mutex
    >;

  using pc_t = typename impl::persist_data<allocator_segment_t, allocator_atomic_t>;

  struct store_root_t
  {
    /* A pointer so that null value can indicate no allocation. */
    PMEMoid pc;
  };

  TOID_DECLARE_ROOT(struct store_root_t);

  class open_session
  {
    TOID(struct store_root_t) _root;
    open_pool_handle          _pop;
    table_t                   _map;
    impl::atomic_controller<table_t> _atomic_state;
    std::string               _dir;
    std::string               _name;
  public:
    explicit open_session(
      TOID(struct store_root_t) &root_
      , open_pool_handle &&pop_
      , const std::string &dir_
      , const std::string &name_
      , pc_t *pc
    )
      : _root(root_)
      , _pop(std::move(pop_))
      , _map(pc, table_t::allocator_type(_pop.get(), type_num::table))
      , _atomic_state(*pc, _map)
      , _dir(dir_)
      , _name(name_)
    {}
    open_session(const open_session &) = delete;
    open_session& operator=(const open_session &) = delete;
    PMEMobjpool *pool() const { return _pop.get(); }
#if 0
    const std::string &path() const noexcept { return _path; }
#endif
    const std::string &dir() const noexcept { return _dir; }
    const std::string &name() const noexcept { return _name; }
    table_t &map() noexcept { return _map; }
    const table_t &map() const noexcept { return _map; }

    auto enter(
      persist_fixed_string<char> &key
      , std::uint64_t type_num_data
      , std::vector<Component::IKVStore::Operation *>::const_iterator first
      , std::vector<Component::IKVStore::Operation *>::const_iterator last
    ) -> Component::status_t
    {
      return _atomic_state.enter(_pop.get(), key, type_num_data, first, last);
    }
  };

  struct tls_cache_t {
    open_session *session;
  };

/* globals */
  thread_local tls_cache_t tls_cache = { nullptr };

  std::mutex sessions_mutex;
  using session_map = std::map<open_session *, std::unique_ptr<open_session>>;
  session_map g_sessions;

  auto locate_session(const IKVStore::pool_t pid) -> open_session &
  {
    auto *const session = reinterpret_cast<struct open_session *>(pid);
    if ( session != tls_cache.session )
    {
      std::unique_lock<std::mutex> sessions_lk(sessions_mutex);
      auto ps = g_sessions.find(session);
      if ( ps == g_sessions.end() )
      {
        throw API_exception(PREFIX "invalid pool identifier %p", __func__, session);
      }
      tls_cache.session = ps->second.get();
    }
    return *tls_cache.session;
  }

  auto move_session(const IKVStore::pool_t pid) -> std::unique_ptr<open_session>
  {
    auto *const session = reinterpret_cast<struct open_session *>(pid);

    std::unique_lock<std::mutex> sessions_lk(sessions_mutex);
    auto ps = g_sessions.find(session);
    if ( ps == g_sessions.end() )
    {
      throw API_exception(PREFIX "invalid pool identifier %p", __func__, session);
    }

    if ( session == tls_cache.session ) { tls_cache.session = nullptr; }
    auto s = std::move(ps->second);
    g_sessions.erase(ps);
    return s;
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
          throw General_exception("pmempool_check failed");
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

    perror("pmempool_check_init");
    return -1;
  }
}

hstore::hstore(const std::string & /* owner */, const std::string & /* name */)
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

namespace
{
  using pc_init_arg = std::tuple<std::size_t>;
  int pc_init(PMEMobjpool *pop, void *ptr_, void *arg_)
  {
    const auto ptr = static_cast<pc_t *>(ptr_);
    const auto arg = static_cast<pc_init_arg *>(arg_);
    new (ptr) pc_t(std::get<0>(*arg), table_t::allocator_type{pop, type_num::table});
    return 0; /* return value is not documented, but might be an error code */
  }

  pc_t *map_create_if_null(
    PMEMobjpool *pop_
    , TOID(struct store_root_t) &root
    , std::size_t expected_obj_count
    , bool verbose
  )
  {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wpedantic"
    if ( OID_IS_NULL(D_RO(root)->pc) )
    {
      if ( verbose )
      {
        PLOG(
          PREFIX "root is empty: new hash required object count %zu"
          , __func__
          , expected_obj_count
        );
      }
      auto oid =
        palloc(
          pop_
          , sizeof(pc_t)
          , type_num::persist
          , pc_init
          , pc_init_arg(expected_obj_count)
          , "persist"
       );
      pc_t *p = static_cast<pc_t *>(pmemobj_direct(oid));
      table_t::allocator_type{pop_, type_num::table}
        .persist(p, sizeof *p, "persist_data");
      D_RW(root)->pc = oid;
    }
    auto rt = D_RW(root);
#pragma GCC diagnostic pop
    auto pc = pmemobj_direct(rt->pc);
    PLOG(PREFIX "persist root addr %p", __func__, static_cast<const void *>(rt));
    return static_cast<pc_t *>(pc);
  }
}

namespace
{
  std::string make_full_path(const std::string &prefix, const std::string &suffix)
  {
    return prefix + ( prefix[prefix.length()-1] != '/' ? "/" : "") + suffix;
  }
}

auto hstore::create_pool(
  const std::string &path,
  const std::string &name,
  const std::size_t size,
  unsigned int /* flags */,
  uint64_t expected_obj_count /* args */) -> pool_t
{
  if ( option_DEBUG )
  {
    PLOG(PREFIX "path=%s pool_name=%s", __func__, path.c_str(), name.c_str());
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  if (PMEMOBJ_MAX_ALLOC_SIZE < size)
  {
    PWRN(
      PREFIX "object too large (max %zu, size %zu)"
      , __func__
      , PMEMOBJ_MAX_ALLOC_SIZE
      , size
    );
#pragma GCC diagnostic pop
    /* NOTE: E_TOO_LARGE may be negative, but pool_t is uint64_t */
    return uint64_t(E_TOO_LARGE);
  }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  if (size < PMEMOBJ_MIN_POOL) {
#pragma GCC diagnostic pop
    PWRN(PREFIX "object too small", __func__);
    /* NOTE: E_BAD_PARAM may be negative, but pool_t is uint64_t */
    return uint64_t(E_BAD_PARAM);
  }

  std::string fullpath = make_full_path(path, name);

  open_pool_handle pop(nullptr, pmemobj_close);

  /* NOTE: conditions can change between the access call and the create/open call.
   * This code makes no prevision for such a change.
   */
  if (access(fullpath.c_str(), F_OK) != 0) {
    if ( option_DEBUG )
    {
      PLOG(PREFIX "creating new pool: %s (%s) size=%lu"
        , __func__
        , name.c_str()
        , fullpath.c_str()
        , size
      );
    }

    boost::filesystem::path p(fullpath);
    boost::filesystem::create_directories(p.parent_path());

    pop.reset(pmemobj_create(fullpath.c_str(), REGION_NAME, size, 0666));
    if (not pop)
    {
      throw
        General_exception("failed to create new pool - %s\n", pmemobj_errormsg());
    }
  }
  else {
    if ( option_DEBUG )
    {
      PLOG(PREFIX "opening existing Pool: %s", __func__, fullpath.c_str());
    }

    if (check_pool(fullpath.c_str()) != 0)
    {
      throw General_exception("pool check failed");
    }

    pop.reset(pmemobj_open(fullpath.c_str(), REGION_NAME));
    if (not pop)
    {
      throw General_exception("failed to re-open pool - %s\n", pmemobj_errormsg());
    }
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
  TOID(struct store_root_t) root = POBJ_ROOT(pop.get(), struct store_root_t);
#pragma GCC diagnostic pop
  assert(!TOID_IS_NULL(root));

  auto pc = map_create_if_null(pop.get(), root, expected_obj_count, option_DEBUG);
  auto session = std::make_unique<open_session>(root, std::move(pop), path, name, pc);
  auto p = session.get();
  std::unique_lock<std::mutex> sessions_lk(sessions_mutex);
  g_sessions.emplace(p, std::move(session));

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
      open_pool_handle(pmemobj_open(fullpath.c_str(), REGION_NAME), pmemobj_close)
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

    auto pc = map_create_if_null(pop.get(), root, 1U, option_DEBUG);
    auto session = std::make_unique<open_session>(root, std::move(pop), path, name, pc);
    auto p = session.get();
    std::unique_lock<std::mutex> sessions_lk(sessions_mutex);
    g_sessions.emplace(p, std::move(session));

    return reinterpret_cast<IKVStore::pool_t>(p);
  }
  throw General_exception("failed to re-open pool - %s\n", pmemobj_errormsg());
}

void hstore::close_pool(const pool_t pid)
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

void hstore::delete_pool(const std::string &dir, const std::string &name)
{
  const int flags = 0;
  if ( auto e = pmempool_rm(make_full_path(dir, name).c_str(), flags) ) {
    throw
      General_exception(
        "unable to delete pool (%s/%s) error %d"
        , dir.c_str()
        , name.c_str()
        , e
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

  auto &session = locate_session(pool);
  auto *const pop = session.pool();
  auto cvalue = static_cast<const char *>(value);

  const auto i =
    session.map().emplace(
      std::piecewise_construct
      , std::forward_as_tuple(key.begin(), key.end(), pop, type_num::key, "key")
      , std::forward_as_tuple(cvalue, cvalue + value_len, pop, type_num::mapped, "value")
    );
  /* ERROR: E_KEY_EXISTS is a guess. It might be that some other key_str
   * which hashes to the same hash key exists.
   */
  return i.second ? S_OK : E_KEY_EXISTS;
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
    auto *const pop = session.pool();
    auto p_key = KEY_T(key.begin(), key.end(), pop, type_num::key, "key");

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
    PWRN("key:%s not found", key.c_str());
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
  auto *const pop = session.pool();
  auto p_key = KEY_T(key.begin(), key.end(), pop, type_num::key, "key");

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
    if ( type == IKVStore::STORE_LOCK_READ ) {
      if ( ! map.lock_shared(p_key) )
      {
        return false;
      }
    }
    else {
      if ( ! map.lock_unique(p_key) )
      {
        return false;
      }
    }
    return true;
  }
}

auto hstore::lock(const pool_t pool,
  const std::string &key,
  lock_type_t type,
  void *& out_value,
  std::size_t & out_value_len) -> key_t
{
  auto &session = locate_session(pool);
  auto *const pop = session.pool();
  const auto p_key = KEY_T(key.begin(), key.end(), pop, type_num::key, "key");

  try
  {
    persist_fixed_string<char> &val = session.map().at(p_key);
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
        , std::forward_as_tuple(out_value_len, pop, type_num::mapped)
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
      auto *const pop = session.pool();
      auto p_key = KEY_T(key->begin(), key->end(), pop, type_num::key, "key");

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
  auto *const pop = session.pool();
  persist_fixed_string<char> *val;
  auto p_key = KEY_T(key.begin(), key.end(), pop, type_num::key, "key");
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
        , std::forward_as_tuple(object_size, pop, type_num::mapped)
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
    auto *const pop = session.pool();
    auto p_key = KEY_T(key.begin(), key.end(), pop, type_num::key, "key");
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
  auto& pop = session.pool();

  HM_CMD(pop, D_RO(root)->map(), cmd, arg);
#endif
}

namespace
{
/* Return value not set. Ignored?? */
  int _functor(
    const std::string &key
    , persist_fixed_string<char> &m
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

  auto p_key = KEY_T(key.begin(), key.end(), session.pool(), type_num::key, "key");

  maybe_lock m(session.map(), p_key, take_lock);

  return session.enter(p_key, type_num::mapped, op_vector.begin(), op_vector.end());
}
catch ( std::exception & )
{
  return E_FAIL;
}

/**
 * Factory entry point
 *
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  return
    component_id == hstore_factory::component_id()
    ? new ::hstore_factory()
    : nullptr
    ;
}
