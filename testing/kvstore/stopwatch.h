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
      start_time = rdtsc(); 
      running = true;
    }
  }

  void stop()
  {
    if (running)
    {
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
  }

  double get_time_in_seconds()
  {
    double running_time = 0;

    if (running) {
      uint64_t stop_time = rdtsc();
      return ((double)(stop_time - start_time)) / cycles_per_second;
    }
    else {
      return ((double)total) / cycles_per_second;
    }
  }

  double get_lap_time()
  {
    return lap_time;
  }


private:
  double total = 0;
  double lap_time = 0;
  bool running = false;
  uint64_t start_time = 0;
  double cycles_per_second = Common::get_rdtsc_frequency_mhz() * 1000000;
};


#endif //  __STOPWATCH_H__
