#ifndef _TEST_MR_LOCK_H_
#define _TEST_MR_LOCK_H_

#include <cstddef> /* size_t */

class mr_lock
{
  const void *_addr;
  std::size_t _len;
public:
  mr_lock(const void *addr, std::size_t len);
  mr_lock(mr_lock &&);
  mr_lock &operator=(mr_lock &&);
  ~mr_lock();
};

#endif
