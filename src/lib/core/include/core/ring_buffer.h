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



/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __CORE_RING_BUFFER_H__
#define __CORE_RING_BUFFER_H__

#include <stdlib.h>
#include <atomic>
#include <string>

struct rte_ring;

namespace Core
{
class Ring_buffer_base {
 private:
  static constexpr bool option_DEBUG = true;

 protected:
  Ring_buffer_base(std::string name, size_t size);
  Ring_buffer_base(std::string name, int index, size_t size);
  virtual ~Ring_buffer_base();

  int full();
  int empty();
  int sp_enqueue(void* elem);
  int sc_dequeue(void*& elem);
  int mp_enqueue(void* elem);
  int mc_dequeue(void*& elem);

 protected:
  static std::atomic<uint64_t> _static_name_index;

  static std::string derive_static_ring_buffer_name() {
    std::string name =
        "rbuf-" + std::to_string(_static_name_index.fetch_add(1));
    return name;
  }

 private:
  struct rte_ring* _rte_ring;
};

/**
 * Ring buffer wrapper for DPDK rte_ring
 *
 */
template <typename T>
class Ring_buffer : public Ring_buffer_base {
 public:
  Ring_buffer(const char* name, size_t size) : Ring_buffer_base(name, size) {
    static_assert(sizeof(T) == sizeof(void*),
                  "Ring_buffer unsupported type (size!=sizeof(void*))");
  }

  Ring_buffer(std::string name, size_t size) : Ring_buffer_base(name, size) {
    static_assert(sizeof(T) == sizeof(void*),
                  "Ring_buffer unsupported type (size!=sizeof(void*))");
  }

  Ring_buffer(const char* name, int index, size_t size)
      : Ring_buffer_base(name, index, size) {
    static_assert(sizeof(T) == sizeof(void*),
                  "Ring_buffer unsupported type (size!=sizeof(void*))");
  }

  Ring_buffer(size_t size)
      : Ring_buffer_base(derive_static_ring_buffer_name(), size) {
    static_assert(sizeof(T) == sizeof(void*),
                  "Ring_buffer unsupported type (size!=sizeof(void*))");
  }

  inline int full() { return Ring_buffer_base::full(); }

  inline int empty() { return Ring_buffer_base::empty(); }

  inline int sp_enqueue(T elem) {
    return Ring_buffer_base::sp_enqueue(reinterpret_cast<void*>(elem));
  }

  inline int sc_dequeue(T& elem) {
    return Ring_buffer_base::sc_dequeue(reinterpret_cast<void*&>(elem));
  }

  inline int mp_enqueue(T elem) {
    return Ring_buffer_base::mp_enqueue(reinterpret_cast<void*>(elem));
  }

  inline int mc_dequeue(T& elem) {
    return Ring_buffer_base::mc_dequeue(reinterpret_cast<void*&>(elem));
  }
};

}  // namespace Core

#endif  // __CORE_RING_BUFFER_H__
