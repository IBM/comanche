#ifndef __STOPWATCH_H__
#define __STOPWATCH_H__

#include <cstdlib>
#include "common/cycles.h"

class Stopwatch
{
public:
  bool is_running()
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

  double get_time_in_seconds()
  {
    assert(!running);
    return ((double)total) / cycles_per_second;
  }

  double get_lap_time_in_seconds()
  {
    return ((double)lap_time) / cycles_per_second;
  }

private:
  uint64_t total = 0;
  uint64_t lap_time = 0;
  uint64_t start_time = 0;
  
  bool     running = false;
  double   cycles_per_second = Common::get_rdtsc_frequency_mhz() * 1000000.0f;
};


#endif //  __STOPWATCH_H__
