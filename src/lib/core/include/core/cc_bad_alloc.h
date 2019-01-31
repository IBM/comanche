#ifndef CORE_BAD_ALLOC
#define CORE_BAD_ALLOC

#include <cstddef> /* size_t */
#include <new> /* bad_alloc */
#include <string>

namespace Core
{
  class CC_bad_alloc
    : public std::bad_alloc
  {
    std::string _what;
  public:
    CC_bad_alloc(std::size_t pad, std::size_t count, std::size_t size)
      : _what(std::string("CC_bad_alloc: ") + std::to_string(pad) + "+" + std::to_string(count) + "*" + std::to_string(size))
    {}
    const char *what() const noexcept override
    {
      return _what.c_str();
    }
  };
}

#endif
