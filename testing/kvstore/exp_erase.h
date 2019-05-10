#ifndef _EXP_ERASE_H_
#define _EXP_ERASE_H_

#include "experiment.h"

#include "statistics.h"

#include <cstdlib>
#include <vector>

class ExperimentErase : public Experiment
{
  std::size_t _i;
  std::vector<double> _start_time;
  std::vector<double> _latencies;
  BinStatistics _latency_stats;

public:
  ExperimentErase(const ProgramOptions &options);

  void initialize_custom(unsigned core) override;
  bool do_work(unsigned core) override;
  void cleanup_custom(unsigned core) override;
};

#endif
