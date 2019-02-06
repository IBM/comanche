#ifndef COMANCHE_HSTORE_ALLOCATOR_H
#define COMANCHE_HSTORE_ALLOCATOR_H

#include "bad_alloc_cc.h"
#include "heap_cc.h"
#include "persister_cc.h"

#include <cstddef> /* size_t, ptrdiff_t */

template <typename T, typename Persister>
  class allocator_cc;

template <>
  class allocator_cc<void, persister>
  {
  public:
    using pointer = void *;
    using const_pointer = const void *;
    using value_type = void;
    template <typename U>
      struct rebind
      {
        using other = allocator_cc<U, persister>;
      };
  };

template <typename Persister>
  class allocator_cc<void, Persister>
  {
  public:
    using pointer = void *;
    using const_pointer = const void *;
    using value_type = void;
    template <typename U>
      struct rebind
      {
        using other = allocator_cc<U, Persister>;
      };
  };

template <typename T, typename Persister = persister>
  class allocator_cc
    : public Persister
  {
    heap_cc _pool;
  public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T &;
    using const_reference = const T &;
    using value_type = T;
    /* we do not have a separate deallocator; use an allocator for deallocation */
    using deallocator_type = allocator_cc<T, Persister>;
    template <typename U>
      struct rebind
      {
        using other = allocator_cc<U, Persister>;
      };

    explicit allocator_cc(void *area_, std::size_t size_, Persister p_ = Persister())
      : Persister(p_)
      , _pool(area_, size_)
    {}

    explicit allocator_cc(void *area_, Persister p_ = Persister())
      : Persister(p_)
      , _pool(area_)
    {}

    allocator_cc(const heap_cc &pool_, Persister p_ = Persister()) noexcept
      : Persister(p_)
      , _pool(pool_)
    {}

    allocator_cc(const allocator_cc &a_) noexcept = default;

    template <typename U, typename P>
      allocator_cc(const allocator_cc<U, P> &a_) noexcept
        : allocator_cc(a_.pool())
      {}

    allocator_cc &operator=(const allocator_cc &a_) = delete;

    pointer address(reference x) const noexcept
    {
      return pointer(&x);
    }
    const_pointer address(const_reference x) const noexcept
    {
      return pointer(&x);
    }

    auto allocate(
      size_type s
      , typename allocator_cc<void, Persister>::const_pointer /* hint */ =
          typename allocator_cc<void, Persister>::const_pointer{}
      , const char * = nullptr
    ) -> pointer
    {
      auto ptr = _pool.malloc(s * sizeof(T));
      if ( ptr == 0 )
      {
        throw bad_alloc_cc(0, s, sizeof(T));
      }
      return static_cast<pointer>(ptr);
    }
    void deallocate(
      pointer p
      , size_type
    )
    {
      _pool.free(p);
    }
    auto max_size() const
    {
      return 8; /* reminder to provide a proper max size value */
    }
    void persist(const void *ptr, size_type len, const char * = nullptr)
    {
      Persister::persist(ptr, len);
    }
    auto pool() const
    {
      return _pool;
    }
  };

#endif
