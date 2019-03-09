#ifndef __EXP_THROUGHPUT_H__
#define __EXP_THROUGHPUT_H__

#include "experiment.h"

#include "statistics.h"
#include "stopwatch.h"

#include <chrono>
#include <cstdlib>
#include <random>
#include <mutex>

class ExperimentThroughput : public Experiment
{
  unsigned long _i_rd;
  unsigned long _i_wr;
  std::chrono::high_resolution_clock::time_point _start_time;
  std::chrono::high_resolution_clock::time_point _report_time;
  std::chrono::high_resolution_clock::duration _report_interval;
  std::default_random_engine _rand_engine;
  std::uniform_int_distribution<unsigned> _rand_pct;
  unsigned _rd_pct;
  unsigned long _op_count_rd;
  unsigned long _op_count_wr;
  unsigned long _op_count_interval_rd;
  unsigned long _op_count_interval_wr;
  static unsigned long _iops;
  static std::mutex _iops_lock;
  Stopwatch _sw_rd;
  Stopwatch _sw_wr;
  bool _continuous;
  static void handler(int);
  static bool _stop;

  std::chrono::high_resolution_clock::duration elapsed(std::chrono::high_resolution_clock::time_point);
  static double to_seconds(std::chrono::high_resolution_clock::duration);

public:
  ExperimentThroughput(const ProgramOptions &options);
  void initialize_custom(unsigned core) override;
  bool do_work(unsigned core) override;
  void cleanup_custom(unsigned core) override;
  static void summarize();
};

#endif //  __EXP_THROUGHPUT_H_
