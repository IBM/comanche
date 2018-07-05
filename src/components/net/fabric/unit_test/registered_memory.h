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
  static constexpr uint64_t remote_key = 54312U; /* value does not matter to ibverbs; ibverbs ignores the value and creates its own key */
  Component::IFabric_connection &_cnxn;
  /* There may be an alignment restriction on registered memory, May be 8, or 16. */
  alignas(64) std::array<char, 4096> _memory;
  registration _registration;
public:
  explicit registered_memory(Component::IFabric_connection &cnxn);

  char &operator[](std::size_t ix) { return _memory[ix]; }
  volatile char &first_char() { return _memory[0]; }

  std::uint64_t key() const { return _registration.key(); }
};

#endif
