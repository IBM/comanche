#ifndef __EXP_THROUGHUPDATE_H__
#define __EXP_THROUGHUPDATE_H__

#include "experiment.h"

#include "statistics.h"
#include "stopwatch.h"

#include <chrono>
#include <cstdlib>
#include <mutex>

extern Data * _data;

class ExperimentThroughupdate : public Experiment
{
  std::chrono::high_resolution_clock::time_point _start_time;
  std::chrono::high_resolution_clock::time_point _end_time;
  std::chrono::high_resolution_clock::time_point _report_time;
  std::chrono::high_resolution_clock::duration _report_interval;
  unsigned long _op_count;
  unsigned long _op_count_interval;
  static unsigned long _iops;
  static std::mutex _iops_lock;
  Stopwatch _sw;
  bool _continuous;
  static void handler(int);
  static bool _stop;

  void initialize_custom(unsigned core) override;
  void cleanup_custom(unsigned core) override;

  std::chrono::high_resolution_clock::duration elapsed(std::chrono::high_resolution_clock::time_point);
  static double to_seconds(std::chrono::high_resolution_clock::duration);

public:
  ExperimentThroughupdate(const ProgramOptions &options);
  bool do_work(unsigned core) override;
  static void summarize();
};

#endif //  __EXP_THROUGHPUT_H_
