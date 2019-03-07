#ifndef __STOPWATCH_H__
#define __STOPWATCH_H__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include "common/cycles.h"
#pragma GCC diagnostic pop

#include <cstddef>
#include <iostream>

class Stopwatch
{
public:
  using duration_t = std::uint64_t;
  using time_point_t = std::uint64_t;
  bool is_running() const
  {
    return running;
  }

  void start()
  {
    if (!running)
    {
      __sync_synchronize(); /* we need the barrier to avoid measuring out of order execution */
      start_time = rdtsc(); 
      running = true;
    }
    else
    {
      std::cerr << "WARNING: trying to start a running counter" << std::endl;
    }
  }

  void stop()
  {
    if (running)
    {
      __sync_synchronize(); /* we need the barrier to avoid measuring out of order execution */
      uint64_t stop_time = rdtsc();
      running = false;

      lap_time = stop_time - start_time;
      total += lap_time; 
    }
  }

  void reset()
  {
    running = false;
    start_time = 0;
    total = 0;
    lap_time = 0;
  }

  double get_time_in_seconds() const
  {
    return double(get_time_native())/cycles_per_second;
  }

  duration_t get_time_native() const
  {
    return
      running
      ? total + (rdtsc() - start_time)
      : total
      ;
  }

  double get_lap_time_in_seconds() const
  {
    return double(lap_time) / cycles_per_second;
  }

  duration_t get_lap_time_native() const
  {
    return lap_time;
  }

  Stopwatch()
    : total()
    , lap_time()
    , start_time()
    , running(false)
    , cycles_per_second(Common::get_rdtsc_frequency_mhz() * 1000000.0)
  {}

private:
  duration_t total;
  duration_t lap_time;
  time_point_t start_time;
  bool     running;
  double   cycles_per_second;
};


#endif //  __STOPWATCH_H__
