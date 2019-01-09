#ifndef __COMMON_RWLOCK_H__
#define __COMMON_RWLOCK_H__

#define _MULTI_THREADED
#include <common/exceptions.h>
#include <pthread.h>

namespace Common
{
class RWLock {
 public:
  RWLock() {
    if (pthread_rwlock_init(&_lock, NULL))
      throw General_exception("unable to initialize RW lock");
  }

  int read_lock() { return pthread_rwlock_rdlock(&_lock); }

  int read_trylock() { return pthread_rwlock_tryrdlock(&_lock); }

  int write_lock() { return pthread_rwlock_wrlock(&_lock); }

  int write_trylock() { return pthread_rwlock_trywrlock(&_lock); }

  int unlock() { return pthread_rwlock_unlock(&_lock); }

  ~RWLock() { pthread_rwlock_destroy(&_lock); }

 private:
  pthread_rwlock_t _lock;
};

class RWLock_guard {
 public:
  enum {
    WRITE = 0,
    READ = 1,
  };

 public:
  RWLock_guard(RWLock &lock, int mode = READ) : _lock(lock) {
    int rc;
    if (mode == WRITE) {
      if (_lock.write_lock() != 0)
        throw General_exception("failed to take write lock");
    }
    else if (mode == READ) {
      if (_lock.read_lock() != 0)
        throw General_exception("failed to take read lock");
    }
    else
      throw API_exception("unexpected RWLock_guard mode (mode=%d)", mode);
  }

  ~RWLock_guard() noexcept(false) {
    if (_lock.unlock() != 0)
      throw General_exception("failed to release lock in guard");
  }

 private:
  RWLock &_lock;
};
}  // namespace Common

#endif  // __COMMON_RWLOCK_H__
