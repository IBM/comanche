#ifndef _TEST_PINGPONG_STAT_H_
#define _TEST_PINGPONG_STAT_H_

#include <chrono> /* high_resolution_clock */
#include <cstdint> /* uint64_t */

class pingpong_stat
{
  std::chrono::high_resolution_clock::time_point _start;
  std::chrono::high_resolution_clock::time_point _stop;
  std::uint64_t _poll_count;
public:
  pingpong_stat()
    : _start(std::chrono::high_resolution_clock::time_point::min())
    , _stop()
    , _poll_count(0U)
  {}
  void do_start() { _start = std::chrono::high_resolution_clock::now(); }
  void do_stop(std::uint64_t poll_count)
  {
    _stop = std::chrono::high_resolution_clock::now();
    _poll_count = poll_count;
  }
  std::chrono::high_resolution_clock::time_point start() const { return _start; }
  std::chrono::high_resolution_clock::time_point stop() const { return _stop; }
  std::uint64_t poll_count() const { return _poll_count; }
};

#endif
