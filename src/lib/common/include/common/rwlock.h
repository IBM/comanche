#ifndef __COMMON_RWLOCK_H__
#define __COMMON_RWLOCK_H__

#include <pthread.h>

namespace Common
{
class RWLock
{
public:
  RWLock() {
    pthread_rwlock_init(&_lock, NULL); 
  }

  int read_lock() {
    return pthread_rwlock_rdlock(&_lock);
  }

  int read_trylock() {
    return pthread_rwlock_tryrdlock(&_lock);
  }

  int write_lock() {
    return pthread_rwlock_wrlock(&_lock);
  }

  int write_trylock() {
    return pthread_rwlock_trywrlock(&_lock);
  }
  
  int unlock() {
    return pthread_rwlock_unlock(&_lock);
  }

  ~RWLock() {
    pthread_rwlock_destroy(&_lock);
  }
  
private:
  pthread_rwlock_t _lock;
};

}

#endif // __COMMON_RWLOCK_H__
