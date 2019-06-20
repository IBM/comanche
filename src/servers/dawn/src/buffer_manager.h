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
#ifndef __BUFFER_MGR_H__
#define __BUFFER_MGR_H__

#ifdef __cplusplus

#include <api/components.h>
#include <api/fabric_itf.h>
#include <api/kvstore_itf.h>
#include <sys/mman.h>
#include "dawn_config.h"

namespace Dawn
{
template <class Transport>
class Buffer_manager {
  static constexpr bool option_DEBUG = false;

 public:
  static constexpr size_t DEFAULT_BUFFER_COUNT = 8;
  static constexpr size_t BUFFER_LEN           = MiB(2);

  enum {
    BUFFER_FLAGS_EXTERNAL = 1,
  };

  using memory_region_t = typename Transport::memory_region_t;

  struct buffer_t {
    iovec *         iov;
    memory_region_t region;
    void *          desc;
    const size_t    original_length;
    int             flags;
    const unsigned  magic;

    buffer_t(size_t length) : original_length(length), flags(0), magic(0xC0FFEE)
    {
    }

    ~buffer_t() { delete iov; }

    inline void * base() { return iov->iov_base; }
    inline size_t length() { return iov->iov_len; }
    inline void   set_length(size_t s) { iov->iov_len = s; }
    inline void   set_external() { flags |= BUFFER_FLAGS_EXTERNAL; }
    inline bool   is_external() const { return flags & BUFFER_FLAGS_EXTERNAL; }
    inline void   reset_length()
    {
      assert(!is_external());
      assert(original_length > 1);
      iov->iov_len = original_length;
    }
    inline bool check_magic() const { return magic == 0xC0FFEE; }
  };

 public:
  Buffer_manager(Transport *transport,
                 size_t     buffer_count = DEFAULT_BUFFER_COUNT)
      : _buffer_count(buffer_count), _transport(transport)
  {
    init();
  }

  ~Buffer_manager()
  {
    for (auto b : _buffers) {
      ::free(b->iov->iov_base);
      delete b;
    }
  }

  buffer_t *allocate()
  {
    if (unlikely(_free.empty())) throw Program_exception("buffer_manager.h: no shard buffers remaining");
    auto iob = _free.back();
    assert(iob->flags == 0);
    _free.pop_back();
    if (option_DEBUG) PLOG("bm: allocate : %p %lu", iob, _free.size());
    return iob;
  }

  void free(buffer_t *iob)
  {
    assert(iob);
    assert(iob->flags == 0);

    if (option_DEBUG) PLOG("bm: free     : %p", iob);
    iob->reset_length();
    _free.push_back(iob);
  }

 private:
  void init()
  {
    auto alloc_iov = []() -> iovec * {
      iovec *iov    = new iovec;
      iov->iov_base = aligned_alloc(MiB(2), BUFFER_LEN);
      iov->iov_len  = BUFFER_LEN;

      assert(iov->iov_base);
      madvise(iov->iov_base, iov->iov_len, MADV_HUGEPAGE);
      memset(iov->iov_base, 0, iov->iov_len);
      return iov;
    };

    for (unsigned i = 0; i < _buffer_count; i++) {
      auto iov = alloc_iov();
      auto region =
          _transport->register_memory(iov->iov_base, iov->iov_len, 0, 0);
      auto desc = _transport->get_memory_descriptor(region);

      auto new_buffer    = new buffer_t(BUFFER_LEN);
      new_buffer->iov    = iov;
      new_buffer->region = region;
      new_buffer->desc   = desc;

      _buffers.push_back(new_buffer);
      _free.push_back(new_buffer);
    }
  }

  using pool_t = Component::IKVStore::pool_t;
  using key_t  = std::uint64_t;

  Transport *             _transport;
  const size_t            _buffer_count;
  std::vector<buffer_t *> _buffers;
  std::vector<buffer_t *> _free;
};

}  // namespace Dawn

#endif
#endif
