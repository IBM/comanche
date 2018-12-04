#ifndef TEST_EYCACATCHER_H_
#define TEST_EYCACATCHER_H_

#include <cstddef> /* size_t */
#include <cstdint> /* uint16_t */

namespace
{
  constexpr char eyecatcher[] = "=====================================";
  constexpr std::size_t remote_memory_offset = 25;
}

#endif
