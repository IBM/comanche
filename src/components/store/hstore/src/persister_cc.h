#ifndef COMANCHE_HSTORE_PERSISTER_H
#define COMANCHE_HSTORE_PERSISTER_H

#include <cstddef> /* size_t */

/* default "persister" for persistent memory: a no-op.
 */
class persister
{
public:
  void persist(const void *, std::size_t) {}
};

#endif
