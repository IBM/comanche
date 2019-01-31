#ifndef CORE_PERSISTER_H
#define CORE_PERSISTER_H

#include <cstddef> /* size_t */

namespace Core
{
  /* default "persister" for persistent memory: a no-op.
   */
  class persister
  {
  public:
    void persist(const void *, std::size_t) {}
  };
}

#endif
