#ifndef __EXP_UPDATE_H__
#define __EXP_UPDATE_H__

#include "experiment.h"

#include "statistics.h"

#include <cstdlib>
#include <vector>

class ExperimentUpdate : public Experiment
{ 
  std::size_t _i;
  std::vector<double> _start_time;
  std::vector<double> _latencies;
  BinStatistics _latency_stats;

public:
  ExperimentUpdate(const ProgramOptions &options);

  void initialize_custom(unsigned core) override;
  bool do_work(unsigned core) override;
  void cleanup_custom(unsigned core) override;
};

#endif //  __EXP_UPDATE_H__
