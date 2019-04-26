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

#ifndef __SPSC_BOUNDED_QUEUE__
#define __SPSC_BOUNDED_QUEUE__

#include <assert.h>
#include <fcntl.h> /* For O_* constants */
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <atomic>
#include <boost/thread.hpp>

#include <common/logging.h>
#include <common/utils.h>
#include "memory.h"

namespace Common
{
/**
 * Single-producer, single-consumer class based on Vyukov's algorithm.
 * This version requires polling during empty conditions.
 *
 */
template <typename T, unsigned QSIZE = 32>
class Spsc_bounded_lfq {
 public:
  /**
   * Constructor
   *
   */
  Spsc_bounded_lfq() : _size(QSIZE), _mask(QSIZE - 1), _head(0), _tail(0) {
    // make sure it's a power of 2
    static_assert((QSIZE != 0) && ((QSIZE & (~QSIZE + 1)) == QSIZE),
                  "bad QSIZE");
    // clear buffer
    __builtin_memset(_buffer, 0, sizeof(_buffer));
  }

  /**
   * Destructor. Will clear up shared memory.
   *
   */
  virtual ~Spsc_bounded_lfq() {}

  /**
   * Enqueue an item
   *
   * @param elem Item to enqueue
   *
   * @return True if enqueued OK. False if blocked, or full.
   */
  bool enqueue(T &elem) {
    const size_t head = _head.load(std::memory_order_relaxed);

    if (((_tail.load(std::memory_order_acquire) - (head + 1)) & _mask) >= 1) {
      _buffer[head & _mask] = elem;
      _head.store(head + 1, std::memory_order_release);
      return true;
    }
    return false;
  }

  /**
   * Dequeue an item
   *
   * @param output Dequeued result
   *
   * @return True on success. False on blocked, or empty.
   */
  bool dequeue(T &output) {
    const size_t tail = _tail.load(std::memory_order_relaxed);

    if (((_head.load(std::memory_order_acquire) - tail) & _mask) >= 1) {
      output = _buffer[_tail & _mask];
      _tail.store(tail + 1, std::memory_order_release);
      return true;
    }
    return false;
  }

  /**
   * Helper to get the base address of buffer
   *
   *
   * @return
   */
  void *buffer_base() const { return (void *) _buffer; }

 private:
  typedef char cache_line_pad_t[128];

  cache_line_pad_t _pad0;
  const size_t _size;
  const size_t _mask;

  cache_line_pad_t _pad1;
  std::atomic<size_t> _head;

  cache_line_pad_t _pad2;
  std::atomic<size_t> _tail;

  cache_line_pad_t _pad3;
  T _buffer[QSIZE] __attribute__((aligned(sizeof(T))));
};

/**
 * Single-producer, single-consumer class based on Vyukov's algorithm.  This
 * version
 * implements a sleep on the consumer when the queue is empty.
 *
 */
template <typename T, unsigned QSIZE = 32>
class Spsc_bounded_lfq_sleeping {
 private:
  enum ConsumerState {
    SLEEPING = 1,
    AWAKE = 2,
  };

  static const int RETRY_THRESHOLD = 5000000;
  static const long TIMEOUT_NS = 10000000;  // 10ms

  Spsc_bounded_lfq<T, QSIZE> _queue __attribute__((aligned(8)));
  bool _exit;
  volatile ConsumerState _consumer_state;
  sem_t _sem;

 public:
  Spsc_bounded_lfq_sleeping() : _queue(), _exit(false), _consumer_state(AWAKE) {
    sem_init(&_sem, 1, 0);  // shared, 0 count
  }

  /**
   * Destructor.  Clears up waker thread.
   *
   */
  ~Spsc_bounded_lfq_sleeping() { sem_destroy(&_sem); }

  /**
   * Enqueue an item
   *
   * @param data Item to enqueue
   *
   * @return True if enqueued OK. False if blocked, or full.
   */
  bool enqueue(T &elem) {
    if (_consumer_state == SLEEPING) {
      sem_post(&_sem);  // wake up waker thread
    }
    return _queue.enqueue(elem); /* we don't sleep on the producer side */
  }

  /**
   * Blocking dequeue
   *
   * @param elem
   *
   * @return
   */
  bool dequeue(T &elem) { return _queue.dequeue(elem); }

  /**
   * Dequeue an object. When the queue is empty, calling threads sleeps
   * until an item is available.
   *
   * @param data Item to copy-dequeue to.
   *
   * @return Return true on success. False on empty queue.
   */
  bool dequeue_sleeping(T &elem) {
    unsigned retries = 0;
    while (!_queue.dequeue(elem)) {
      retries++;

      if (retries < RETRY_THRESHOLD) {
        cpu_relax();
        continue;
      }

      // go to sleep
      _consumer_state = SLEEPING;

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
        while ((s = sem_timedwait(&_sem, &ts)) == -1 && errno == EINTR)
          continue; /* Restart if interrupted by handler */
      }

      if (_exit) return false;
    }
    _consumer_state = AWAKE;
    return true;
  }

  /**
   * Signal deque threads to return even without element
   *
   */
  void _exitthreads() {
    _exit = true;
    sem_post(&_sem);
  }

  /**
   * Debug helper to get base address of queue
   *
   *
   * @return Base address of queue
   */
  void *buffer_base() const { return (void *) _queue.buffer_base(); }

 private:
};

#if 0
/** 
 * Single-producer, single-consumer class based on Vyukov's algorithm.  This version
 * implements a sleep on both the consumer (when empty) and producer (when full)
 * sides. 
 * 
 */
template<typename T>
class Spsc_bound_lfq_sleeping_dual_t 
{
private:
  enum ThreadState {
    SLEEPING = 1,
    AWAKE = 2,
  };

  static const int RETRY_THRESHOLD = 1000;
  static const long TIMEOUT_NS = 10000000; // 10ms

  Spsc_bounded_lfq<T>          queue_ __attribute__((aligned(8)));
  bool                         _exit;
  volatile ThreadState         _consumer_state;
  volatile ThreadState         producer_state_;
  sem_t                        consumer_sem_;
  sem_t                        producer_sem_;

public:

  /** 
   * Constructor requiring boost-compatible shared memory segment
   * 
   * @param size Size of queue in elements 
   * @param segment Boost shared memory segment to use.  
   * Note, this should be mapped to the same virtual address on either side.
   *
   */
  Spsc_bound_lfq_sleeping_dual_t(size_t size,
                                 boost::interprocess::fixed_managed_shared_memory * segment) 
    : queue_(size, segment),
      _exit(false),
      _consumer_state(AWAKE),
      producer_state_(AWAKE)
  {
    sem_init(&consumer_sem_,1,0); // shared, 0 count
    sem_init(&producer_sem_,1,0); // shared, 0 count
  }


  /** 
   * Destructor.  Clears up waker thread.
   * 
   */
  ~Spsc_bound_lfq_sleeping_dual_t() {
    sem_destroy(&consumer_sem_);
    sem_destroy(&producer_sem_);
  }
    

  /** 
   * Enqueue an item
   * 
   * @param data Item to enqueue
   * 
   * @return True if enqueued OK. False if blocked, or full.
   */
  bool enqueue(T& elem) {
    if(_consumer_state == SLEEPING) {
      sem_post(&consumer_sem_); // wake up waker thread 
    }
      
    unsigned retries = 0;
    while(!queue_.enqueue(elem)) {
      if(retries < RETRY_THRESHOLD) {
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
        ts.tv_nsec += TIMEOUT_NS; // lets not worry about wrap arounds, shorter timeout is ok

        int s;
        while ((s = sem_timedwait(&producer_sem_, &ts)) == -1 && errno == EINTR)
          continue;       /* Restart if interrupted by handler */
      }

    }

    if(_exit) return false;

    producer_state_ = AWAKE;

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
  bool dequeue(T& elem) {

    if(producer_state_ == SLEEPING) {
      sem_post(&producer_sem_); // wake up waker thread 
    }

    unsigned retries = 0;
    while(!queue_.dequeue(elem)) {
      retries++;

      if(retries < RETRY_THRESHOLD) {
        cpu_relax();
        continue;
      }

      // go to sleep 
      _consumer_state = SLEEPING;

      // timed semaphore to avoid race-condition
      {
        struct timespec ts;
        if (clock_gettime(CLOCK_REALTIME, &ts) == -1) {
          PERR("bad clock_gettime call");
          assert(0);
          return false;
        }
        ts.tv_nsec += TIMEOUT_NS; // lets not worry about wrap arounds, shorter timeout is ok

        int s;
        while ((s = sem_timedwait(&consumer_sem_, &ts)) == -1 && errno == EINTR)
          continue;       /* Restart if interrupted by handler */
      }

      if(_exit) return false;
    }
    _consumer_state = AWAKE;
    return true;
  }

  /** 
   * Signal deque threads to return even without element
   * 
   */
  void _exitthreads() {
    _exit = true;
    sem_post(&producer_sem_);
    sem_post(&consumer_sem_);
  }

private:    

};
#endif
}  // namespace Common

#endif
