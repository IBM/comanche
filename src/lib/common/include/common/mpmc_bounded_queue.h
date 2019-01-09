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

// Note:
// A combination of the algorithms described by the circular buffers
// documentation found in the Linux kernel, and the bounded MPMC queue
// by Dmitry Vyukov[1]. Implemented in pure C++11. Should work across
// most CPU architectures.
//

#ifndef __MPMC_BOUNDED_QUEUE__
#define __MPMC_BOUNDED_QUEUE__

#include <assert.h>
#include <atomic>

#include <common/utils.h>
#include <semaphore.h>
#include <sstream>
#include "memory.h"

#define DCACHE1_LINESIZE 64
#define __cacheline_aligned __attribute__((aligned(DCACHE1_LINESIZE)))

namespace Common
{
/**
 * Multi-producer, multi-consumer class based on Vyukov's algorithm.  This
 * version
 * requires polling during empty conditions.
 *
 */
template <typename T = void *>
class Mpmc_bounded_lfq {
 private:
  struct node_t {
    volatile T data;
    volatile std::atomic<size_t> seq;
  };

 public:
  /**
   * Constructor requiring generic memory allocator
   *
   * @param size Size of queue in elements.
   * @param allocator Generic shared memory allocator.
   *
   */
  Mpmc_bounded_lfq(size_t size, Base_memory_allocator *allocator)
      : _size(size),
        _mask(size - 1),
        _buffer(NULL),
        _head_seq(0),
        _tail_seq(0),
        _allocator(allocator) {
    assert(allocator);

    std::stringstream ss;
    ss << this;
    _memory_id = ss.str();

    _buffer = reinterpret_cast<node_t *>(
        new (_allocator->alloc(sizeof(aligned_node_t) * _size, -1 /*numa*/, 64))
            aligned_node_t[_size]);
    assert(_buffer);

    // make sure it's a power of 2
    assert((_size != 0) && ((_size & (~_size + 1)) == _size));

    // populate the sequence initial values
    for (size_t i = 0; i < _size; ++i) {
      _buffer[i].seq.store(i, std::memory_order_relaxed);
    }
  }

  /**
   * Constructor requiring buffer address
   *
   * @param size Size of queue in elements.
   * @param buf_addr The actual buffer array starting address.
   *
   */
  Mpmc_bounded_lfq(size_t size, void *buf_addr)
      : _size(size),
        _mask(size - 1),
        _buffer(NULL),
        _head_seq(0),
        _tail_seq(0),
        _allocator(NULL) {
    assert(buf_addr);

    std::stringstream ss;
    ss << this;
    _memory_id = ss.str();

    PINF("mpmc size=%ld buffer=%p", size, buf_addr);

    assert(check_aligned(buf_addr, sizeof(aligned_node_t)));
    _buffer = reinterpret_cast<node_t *>(new (buf_addr) aligned_node_t[_size]);
    assert(_buffer);

    assert((_size != 0) && ((_size & (~_size + 1)) ==
                            _size));  // make sure queue len is a power of 2

    // populate the sequence initial values
    for (size_t i = 0; i < _size; ++i) {
      _buffer[i].seq.store(i, std::memory_order_relaxed);
    }
  }

  ~Mpmc_bounded_lfq() {
    if (_allocator) {
      for (size_t i = 0; i < _size; ++i)
        ((aligned_node_t *) &_buffer[i])
            ->~aligned_node_t(); /* manually call dtors */

      _allocator->free(_buffer);
    }
  }

  void exit_threads() {}

  /**
   * Returns a shared memory pointer to the allocated buffer. This can be
   * used to free the memory after destruction of the object
   *
   *
   * @return Pointer to allocated buffer
   */
  node_t *allocated_memory() const { return _buffer; }

  /**
   * Enqueue an item
   *
   * @param data Item to enqueue
   *
   * @return True if enqueued OK. False if blocked, or full.
   */
  bool enqueue(const T &data) {
    // _head_seq only wraps at MAX(_head_seq) instead we use a mask to convert
    // the sequence to an array index
    // this is why the ring buffer must be a size which is a power of 2. this
    // also allows the sequence to double as a ticket/lock.
    size_t head_seq = _head_seq.load(std::memory_order_relaxed);

    for (;;) {
      node_t *node = &_buffer[head_seq & _mask];
      size_t node_seq = node->seq.load(std::memory_order_acquire);
      intptr_t dif = (intptr_t) node_seq - (intptr_t) head_seq;

      // if seq and head_seq are the same then it means this slot is empty
      if (dif == 0) {
        // claim our spot by moving head
        // if head isn't the same as we last checked then that means someone
        // beat us to the punch
        // weak compare is faster, but can return spurious results
        // which in this instance is OK, because it's in the loop
        if (_head_seq.compare_exchange_weak(head_seq, head_seq + 1,
                                            std::memory_order_relaxed)) {
          // set the data
          node->data = data;
          // increment the sequence so that the tail knows it's accessible
          node->seq.store(head_seq + 1, std::memory_order_release);
          return true;
        }
      } else if (dif < 0) {
        // if seq is less than head seq then it means this slot is full
        // and therefore the buffer is full
        return false;
      } else {
        // under normal circumstances this branch should never be taken
        head_seq = _head_seq.load(std::memory_order_relaxed);
      }
    }

    // never taken
    return false;
  }

  /**
   * Dequeue an object.
   *
   * @param data Item to copy-dequeue to.
   *
   * @return Return true on success. False on empty queue.
   */
  bool dequeue(T &data) {
    size_t tail_seq = _tail_seq.load(std::memory_order_relaxed);

    for (;;) {
      node_t *node = &_buffer[tail_seq & _mask];
      size_t node_seq = node->seq.load(std::memory_order_acquire);
      intptr_t dif = (intptr_t) node_seq - (intptr_t)(tail_seq + 1);

      // if seq and head_seq are the same then it means this slot is empty
      if (dif == 0) {
        // claim our spot by moving head
        // if head isn't the same as we last checked then that means someone
        // beat us to the punch
        // weak compare is faster, but can return spurious results
        // which in this instance is OK, because it's in the loop
        if (_tail_seq.compare_exchange_weak(tail_seq, tail_seq + 1,
                                            std::memory_order_relaxed)) {
          // set the output
          data = node->data;
          // set the sequence to what the head sequence should be next time
          // around
          node->seq.store(tail_seq + _mask + 1, std::memory_order_release);
          return true;
        }
      } else if (dif < 0) {
        // queue is empty
        return false;
      } else {
        // under normal circumstances this branch should never be taken
        tail_seq = _tail_seq.load(std::memory_order_relaxed);
      }
    }

    // never taken
    return false;
  }

  /* compatibility helpers */

  inline bool push(const T &data) { return enqueue(data); }

  inline bool pop(T &data) { return dequeue(data); }

  bool empty() {
    size_t tail_seq = _tail_seq.load(std::memory_order_relaxed);

    for (;;) {
      node_t *node = &_buffer[tail_seq & _mask];
      size_t node_seq = node->seq.load(std::memory_order_acquire);
      intptr_t dif = (intptr_t) node_seq - (intptr_t)(tail_seq + 1);

      // if seq and head_seq are the same then it means this slot is empty
      if (dif == 0) {
        return false;
      } else if (dif < 0) {
        // queue is empty
        return true;
      } else {
        // under normal circumstances this branch should never be taken
        tail_seq = _tail_seq.load(std::memory_order_relaxed);
      }
    }

    // never taken
    return false;
  }

  static size_t memory_footprint(size_t queue_size) {
    return sizeof(Common::Mpmc_bounded_lfq<void *>) +
           (sizeof(Common::Mpmc_bounded_lfq<void *>::aligned_node_t) *
            queue_size);
  }

 public:
  typedef typename std::aligned_storage<
      sizeof(node_t), std::alignment_of<node_t>::value>::type aligned_node_t;

 private:
  std::atomic<size_t> _head_seq __cacheline_aligned;
  std::atomic<size_t> _tail_seq __cacheline_aligned;

  byte _padding[DCACHE1_LINESIZE];
  std::string _memory_id;
  Base_memory_allocator *_allocator = nullptr;
  const size_t _size;
  const size_t _mask;
  node_t *_buffer;
};

/**
 * Multi-producer, multi-consumer class based on Vyukov's algorithm.
 * This version implements a sleep on the consumer when the queue is
 * empty.
 *
 */
template <typename T>
class Mpmc_bounded_lfq_sleeping {
 private:
  enum ConsumerState {
    SLEEPING = 1,
    AWAKE = 2,
  };

  static const int RETRY_THRESHOLD = 100000;  // # of retries before sleeping
  static const long TIMEOUT_NS = 1000000;     // 1ms

  Mpmc_bounded_lfq<T> queue_;
  bool exit_;
  sem_t sem_;
  volatile ConsumerState consumer_state_;

 public:
  Mpmc_bounded_lfq_sleeping(size_t size, Base_memory_allocator *allocator)
      : queue_(size, allocator), exit_(false), consumer_state_(AWAKE) {
    sem_init(&sem_, 1, 0);  // shared, 0 count
  }

  Mpmc_bounded_lfq_sleeping(size_t size, void *buf_addr)
      : queue_(size, buf_addr), exit_(false), consumer_state_(AWAKE) {
    sem_init(&sem_, 1, 0);  // shared, 0 count
  }

  /**
   * Destructor. Clears up waker thread.
   *
   */
  ~Mpmc_bounded_lfq_sleeping() {
    exit_ = true;
    sem_post(&sem_);
    sem_destroy(&sem_);
  }

  /**
   * Enqueue an item
   *
   * @param data Item to enqueue
   *
   * @return True if enqueued OK. False if blocked, or full.
   */
  bool enqueue(const T &elem) {
    if (consumer_state_ == SLEEPING) sem_post(&sem_);  // wake up waker thread

    return queue_.enqueue(elem); /* we don't sleep on the producer side */
  }

  /**
   * Dequeue an object. When the queue is empty, calling threads sleeps
   * until an item is available.
   *
   * @param data Item to copy-dequeue to.
   *
   * @return Return true on success. False on empty queue.
   */
  bool dequeue(T &elem) {
    unsigned retries = 0;
    while (!queue_.dequeue(elem)) {
      retries++;
      if (retries < RETRY_THRESHOLD) continue;

      // go to sleep
      consumer_state_ = SLEEPING;

      // timed semaphore to avoid race-condition
      {
        struct timespec ts;
        if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
          PERR("bad clock_gettime call");
          assert(0);
          return false;
        }
        ts.tv_nsec += TIMEOUT_NS;  // lets not worry about wrap arounds, shorter
                                   // timeout is ok

        int s;
        while ((s = sem_timedwait(&sem_, &ts)) == -1 && errno == EINTR)
          continue; /* Restart if interrupted by handler */
      }

      if (exit_) return false;
    }
    consumer_state_ = AWAKE;
    return true;
  }

  /**
   * Signal deque threads to return even without element
   *
   */
  void exit_threads() {
    exit_ = true;
    sem_post(&sem_);
  }

  inline bool empty() { return queue_.empty(); }

  static size_t memory_footprint(size_t queue_size) {
    return sizeof(Common::Mpmc_bounded_lfq<void *>) +
           (sizeof(Common::Mpmc_bounded_lfq<void *>::aligned_node_t) *
            queue_size);
  }
};

/**
 * Multi-producer, multi-consumer class based on Vyukov's algorithm.  This
 * version
 * implements a sleep on the consumer when the queue is empty.
 *
 */
template <typename T>
class Mpmc_bounded_lfq_sleeping_dual {
 private:
  enum ThreadState {
    SLEEPING = 1,
    AWAKE = 2,
  };

  static const int RETRY_THRESHOLD = 1000;  // # of retries before dual_sleeping
  static const long TIMEOUT_NS = 10000000;  // 10ms

  Mpmc_bounded_lfq<T> queue_;
  bool exit_;

  sem_t consumer_sem_;
  sem_t producer_sem_;
  ThreadState consumer_state_;
  ThreadState producer_state_;

 public:
  Mpmc_bounded_lfq_sleeping_dual(size_t size, Base_memory_allocator *allocator)
      : queue_(size, allocator),
        exit_(false),
        consumer_state_(AWAKE),
        producer_state_(AWAKE) {
    sem_init(&consumer_sem_, 1, 0);  // shared, 0 count
    sem_init(&producer_sem_, 1, 0);  // shared, 0 count
  }

  Mpmc_bounded_lfq_sleeping_dual(size_t size, void *buf_addr)
      : queue_(size, buf_addr),
        exit_(false),
        consumer_state_(AWAKE),
        producer_state_(AWAKE) {
    sem_init(&consumer_sem_, 1, 0);  // shared, 0 count
    sem_init(&producer_sem_, 1, 0);  // shared, 0 count
  }

  /**
   * Destructor. Clears up waker thread.
   *
   */
  ~Mpmc_bounded_lfq_sleeping_dual() {
    exit_ = true;
    sem_post(&producer_sem_);
    sem_post(&consumer_sem_);
    sem_destroy(&producer_sem_);
    sem_destroy(&consumer_sem_);
  }

  /**
   * Enqueue an item
   *
   * @param data Item to enqueue
   *
   * @return True if enqueued OK. False if blocked, or full.
   */
  bool enqueue(T &elem) {
    if (consumer_state_ == SLEEPING) sem_post(&consumer_sem_);

    unsigned retries = 0;
    while (!queue_.enqueue(elem)) {
      if (retries < RETRY_THRESHOLD) {
        cpu_relax();
        continue;
      }

      producer_state_ = SLEEPING;

      // timed semaphore to avoid race-condition
      {
        struct timespec ts;
        if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
          PERR("bad clock_gettime call");
          assert(0);
          return false;
        }
        ts.tv_nsec += TIMEOUT_NS;  // lets not worry about wrap arounds, shorter
                                   // timeout is ok

        int s;
        while ((s = sem_timedwait(&producer_sem_, &ts)) == -1 && errno == EINTR)
          continue; /* Restart if interrupted by handler */
      }
    }
    return true;
  }

  /**
   * Dequeue an object. When the queue is empty, calling threads sleeps
   * until an item is available.
   *
   * @param data Item to copy-dequeue to.
   *
   * @return Return true on success. False on empty queue.
   */
  bool dequeue(T &elem) {
    if (producer_state_ == SLEEPING) {
      sem_post(&producer_sem_);  // wake up waker thread
    }

    unsigned retries = 0;
    while (!queue_.dequeue(elem)) {
      retries++;
      if (retries < RETRY_THRESHOLD) continue;

      // go to sleep
      consumer_state_ = SLEEPING;

      // timed semaphore to avoid race-condition
      {
        struct timespec ts;
        if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
          PERR("bad clock_gettime call");
          assert(0);
          return false;
        }
        ts.tv_nsec += TIMEOUT_NS;  // lets not worry about wrap arounds, shorter
                                   // timeout is ok

        int s;
        while ((s = sem_timedwait(&consumer_sem_, &ts)) == -1 && errno == EINTR)
          continue; /* Restart if interrupted by handler */
      }

      if (exit_) return false;
    }
    consumer_state_ = AWAKE;
    return true;
  }

  /**
   * Signal deque threads to return even without element
   *
   */
  void exit_threads() {
    exit_ = true;
    sem_post(&producer_sem_);
    sem_post(&consumer_sem_);
  }

  inline bool empty() { return queue_.empty(); }

  static size_t memory_footprint(size_t queue_size) {
    return sizeof(Common::Mpmc_bounded_lfq<void *>) +
           (sizeof(Common::Mpmc_bounded_lfq<void *>::aligned_node_t) *
            queue_size);
  }
};
}  // namespace Common

#endif
