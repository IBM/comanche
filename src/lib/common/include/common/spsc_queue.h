/*
   eXokernel Development Kit (XDK)

   Samsung Research America Copyright (C) 2013

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.
   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.

   As a special exception, if you link the code in this file with
   files compiled with a GNU compiler to produce an executable,
   that does not cause the resulting executable to be covered by
   the GNU Lesser General Public License.  This exception does not
   however invalidate any other reasons why the executable file
   might be covered by the GNU Lesser General Public License.
   This exception applies to code released by its copyright holders
   in files containing the exception.
*/
// This is free and unencumbered software released into the public domain.

// Anyone is free to copy, modify, publish, use, compile, sell, or
// distribute this software, either in source code form or as a compiled
// binary, for any purpose, commercial or non-commercial, and by any
// means.

// In jurisdictions that recognize copyright laws, the author or authors
// of this software dedicate any and all copyright interest in the
// software to the public domain. We make this dedication for the benefit
// of the public at large and to the detriment of our heirs and
// successors. We intend this dedication to be an overt act of
// relinquishment in perpetuity of all present and future rights to this
// software under copyright law.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

// For more information, please refer to <http://unlicense.org/>

#ifndef __COMMON_SPSC_QUEUE__
#define __COMMON_SPSC_QUEUE__

#include <atomic>

namespace Common {
/**
 * Non-intrusive lock-free unbounded single-consumer/single-producer (SPSC)
 * queue.
 * The algorithm was taken from the blog post below, and converted to C++11.
 * Source:
 * http://cbloomrants.blogspot.com/2009/02/02-26-09-low-level-threading-part-51.html
 */
template <typename T>
class spsc_queue_t
{
 public:
  /** Constructor. */
  spsc_queue_t() : _head(reinterpret_cast<node_t*>(new node_aligned_t)), _tail(_head)
  {
    _head->next = NULL;
  }

  /** Destructor. */
  virtual ~spsc_queue_t()
  {
    T output;
    while (this->dequeue(output)) {
    }
    delete _head;
  }

  /** Inserts an item into the back of the queue. */
  void enqueue(const T& input)
  {
    node_t* node = reinterpret_cast<node_t*>(new node_aligned_t);
    node->data   = input;
    node->next   = NULL;

    std::atomic_thread_fence(std::memory_order_acq_rel);
    _head->next = node;
    _head       = node;
  }

  /** Gets an item from the front of the queue. */
  bool dequeue(T& output)
  {
    std::atomic_thread_fence(std::memory_order_consume);
    if (!_tail->next) {
      return false;
    }

    output = _tail->next->data;
    std::atomic_thread_fence(std::memory_order_acq_rel);
    _back = _tail;
    _tail = _back->next;

    delete _back;
    return true;
  }

 private:
  /** Internal list node. */
  struct node_t {
    node_t* next;
    T       data;
  };

  typedef typename std::aligned_storage<sizeof(node_t), std::alignment_of<node_t>::value>::type
    node_aligned_t;

  node_t* _head;
  char    _cache_line[64];
  node_t* _tail;
  node_t* _back;

  // Private copy constructor and =operator.
  spsc_queue_t(const spsc_queue_t&)
  {
  }
  void operator=(const spsc_queue_t&)
  {
  }
};
}

#endif
