#ifndef DAWN_HSTORE_PERSIST_FIXED_STRING_H
#define DAWN_HSTORE_PERSIST_FIXED_STRING_H

#include "palloc.h"
#include "persistent.h"
#include "pobj_pointer.h"
#include <libpmemobj.h> /* pmemobj_direct */

#include <algorithm>
#include <cstddef> /* size_t */
#include <tuple>

template <typename T>
  class persist_fixed_string;

template <typename T>
  union rep;

template <typename T>
  class fixed_string_access
  {
    fixed_string_access() {}
 public:
    friend rep<T>;
  };

/*
 * - fixed_string
 * - ref_count: because the object may be referenced twice as the table expands
 * - size: length of data (data immediately follows the fixed_string object)
 */
template <typename T>
  class fixed_string
  {
    unsigned _ref_count;
    uint64_t _size;
  public:
    using access = fixed_string_access<T>;
    template <typename IT>
      fixed_string(IT first_, IT last_, access a_)
        : fixed_string((last_-first_), a_)
      {
        /* for small lengths we copy */
        std::copy(first_, last_, static_cast<T *>(static_cast<void *>(this+1)));
      }
    fixed_string(std::size_t data_len_, access)
      : _ref_count(1U)
      , _size(data_len_)
    {
    }
    uint64_t size(access) const noexcept { return _size; }
    unsigned inc_ref(access) noexcept { return _ref_count++; }
    unsigned dec_ref(access) noexcept { return --_ref_count; }
  };

template <typename T>
  union rep
  {
    using element_type = fixed_string<T>;
    using ptr_t = persistent_t<pobj_pointer<element_type>>;
    using access = fixed_string_access<T>;

    template <typename IT>
      using new_pfs_arg = std::tuple<IT, IT>;

    template <typename IT>
      static int new_pfs(PMEMobjpool *, void *ptr_, void *arg_)
      {
        const auto ptr = static_cast<element_type *>(ptr_);
        const auto arg = static_cast<new_pfs_arg<IT> *>(arg_);
        new (ptr) element_type(std::get<0>(*arg), std::get<1>(*arg), access{});
        /* return value is undocumented, but might be an error code */
        return 0;
      }

    using new_pfs_arg_1 = std::tuple<std::size_t>;

    static int new_pfs_1(PMEMobjpool *, void *ptr_, void *arg_)
    {
      const auto ptr = static_cast<element_type *>(ptr_);
      const auto arg = static_cast<new_pfs_arg_1 *>(arg_);
      new (ptr) element_type(std::get<0>(*arg), access{});
      /* return value is undocumented, but might be an error code */
      return 0;
    }

    struct small_t
    {
      char value[23];
      std::uint8_t _size; /* discriminant */
      static constexpr uint8_t large_kind = sizeof value + 1;
      small_t(std::size_t size_)
        : _size(size_ <= sizeof value ? uint8_t(size_) : large_kind) /* note: as of C++17, can use std::clamp */
      {}
    } small;

    ptr_t ptr;
    static_assert( sizeof ptr <= sizeof small.value, "_ptr conflicts with small.size");
    rep()
      : small(0)
    {
    }

    template <typename IT>
      rep(
        IT first_
        , IT last_
        , PMEMobjpool *pop_
        , uint64_t type_num
        , const char *use_
      )
        : small((last_ - first_) * sizeof(T))
      {
        if ( is_small() )
        {
          std::copy(first_, last_, static_cast<T *>(static_cast<void *>(&small.value[0])));
        }
        else
        {
          auto data_size = (last_ - first_) * sizeof(T);
          new (&ptr)
            ptr_t(
              pobj_pointer<element_type>(
                palloc(
                  pop_
                  , sizeof(element_type) + data_size
                  , type_num
                  , new_pfs<IT>
                  , new_pfs_arg<IT>(first_, last_)
                  , use_
                )
              )
            )
          ;
        }
      }

    rep(std::size_t data_len_, PMEMobjpool *pop_, uint64_t type_num)
      : small(data_len_ * sizeof(T))
    {
      /* large data sizes: data not copied, but space is reserved for RDMA */
      if ( is_small() )
      {
      }
      else
      {
        auto data_size = data_len_ * sizeof(T);
        new (&ptr)
          ptr_t(
            pobj_pointer<element_type>(
              palloc(
                pop_
                , sizeof(element_type) + data_size
                , type_num
                , new_pfs_1
                , new_pfs_arg_1(data_len_)
                , "value-space"
              )
            )
          )
        ;
      }
    }

    rep(const rep &other)
      : small(other.small._size)
    {
      if ( is_small() )
      {
        small = other.small;
      }
      else
      {
        small = other.small;
        new (&ptr) ptr_t(other.ptr);
        if ( ptr )
        {
          ptr->inc_ref(access{});
        }
      }
    }

    rep(rep &&other)
      : small(other.small.size)
    {
      if ( is_small() )
      {
        small = other.small;
      }
      else
      {
        small = other.small;
        new (&ptr) ptr_t(other.ptr);
        other.ptr = pobj_pointer<element_type>{};
      }
    }

    rep &operator=(const rep &other)
    {
      if ( other.is_small() )
      {
        small = other.small;
      }
      else
      {
        auto p = ptr;
        small = other.small;
        ptr = other.ptr;
        if ( ptr )
        {
          ptr->inc_ref(access{});
        }

        if ( p )
        {
          if ( p->dec_ref(access{}) == 0 )
          {
            /* Note: in this application (hashed key-value) we never expect
             * to delete a value through operator=, but only through
             * ~persist_fixed_string
             */
            zfree(p, "data operator=");
          }
        }
      }
      return *this;
    }

    rep &operator=(rep &&other)
    {
      if ( other.is_small() )
      {
        small = other.small;
      }
      else
      {
        small = other.small;
        ptr = other.ptr;
        other.ptr = ptr_t();
      }
      return *this;
    }

    ~rep()
    {
      if ( ! is_small() && ptr )
      {
        if ( ptr->dec_ref(access{}) == 0 )
        {
          ptr->~element_type();
          zfree(ptr, "data dtor");
        }
      }
    }

    static constexpr uint8_t large_kind = sizeof small.value + 1;
    bool is_small() const
	{
		assert(small._size <= large_kind);
		return small._size < large_kind;
	}

    std::size_t size() const { return is_small() ? small._size : ptr->size(access{}); }

    const T *data() const
    {
      if ( is_small() )
      {
        return static_cast<const T *>(&small.value[0]);
      }
      auto pt = static_cast<element_type *>(pmemobj_direct(ptr));
      return static_cast<const T *>(static_cast<void *>(pt+1));
    }

    T *data()
    {
      if ( is_small() )
      {
        return static_cast<T *>(&small.value[0]);
      }
      auto pt = static_cast<element_type *>(pmemobj_direct(ptr));
      return static_cast<T *>(static_cast<void *>(pt+1));
    }

  };

template <typename T>
  class persist_fixed_string
  {
    using access = fixed_string_access<T>;
    using element_type = fixed_string<T>;
    using ptr_t = persistent_t<pobj_pointer<element_type>>;
    /* Note: rep is most of persist_fixed_string; it is conceptually its base class.
     * It does not directly replace persist_fixed_string only to preserve the
     * declaration of persist_fixed_string as a class, not a union
     */
    rep<T> _rep;
    /* NOTE: allocating the data string adjacent to the header of a fixed_string
     * precludes use of a standard allocator
     */

  public:
    persist_fixed_string()
      : _rep()
    {
    }

    template <typename IT>
      persist_fixed_string(
        IT first_
        , IT last_
        , PMEMobjpool *pop_
        , uint64_t type_num
        , const char *use_
      )
        : _rep(first_, last_, pop_, type_num, use_)
    {
    }

    persist_fixed_string(std::size_t data_len_, PMEMobjpool *pop_, uint64_t type_num)
      : _rep(data_len_, pop_, type_num)
    {
    }

    persist_fixed_string(const persist_fixed_string &other)
      : _rep(other._rep)
    {
    }

    persist_fixed_string(persist_fixed_string &&other)
      : _rep(other._rep)
    {
    }

    persist_fixed_string &operator=(const persist_fixed_string &other) = default;
    persist_fixed_string &operator=(persist_fixed_string &&other) = default;

    ~persist_fixed_string()
    {
    }

    std::size_t size() const { return _rep.size(); }

    const T *data() const { return _rep.data(); }

    T *data() { return _rep.data(); }
  };

template <typename T>
  bool operator==(
    const persist_fixed_string<T> &a
    , const persist_fixed_string<T> &b
  )
  {
    return
      a.size() == b.size()
      &&
      std::equal(a.data(), a.data() + a.size(), b.data())
    ;
  }

#endif
