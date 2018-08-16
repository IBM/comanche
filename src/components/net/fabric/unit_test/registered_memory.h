#ifndef _TEST_REGISTERED_MEMORY_H_
#define _TEST_REGISTERED_MEMORY_H_

#include "registration.h"
#include <array>
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */

namespace Component
{
  class IFabric_connection;
}

class registered_memory
{
  Component::IFabric_connection &_cnxn;
  /* There may be an alignment restriction on registered memory, May be 8, or 16. */
  alignas(64) std::array<char, 4096> _memory;
  registration _registration;
public:
  /*
   * NOTE: if the memory remote key is used (that is, if the mr attributes do not include FI_PROV_KEY),
   * the key must the unique among registered memories.
   */
  explicit registered_memory(Component::IFabric_connection &cnxn, std::uint64_t remote_key);

  char &operator[](std::size_t ix) { return _memory[ix]; }
  volatile char &first_char() { return _memory[0]; }

  std::uint64_t key() const { return _registration.key(); }
  void *desc() const { return _registration.desc(); }
};

#endif
