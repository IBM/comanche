#ifndef TEST_EYCACATCHER_H_
#define TEST_EYCACATCHER_H_

#include <cstddef> /* size_t */
#include <cstdint> /* uint16_t */

namespace
{
  constexpr std::uint16_t control_port_0 = 47591;
  constexpr std::uint16_t control_port_1 = 47592;
  constexpr std::uint16_t control_port_2 = 47593;
  constexpr char eyecatcher[] = "=====================================";
  constexpr std::size_t remote_memory_offset = 25;
}

#endif
