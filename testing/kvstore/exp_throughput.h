#ifndef __EXP_THROUGHPUT_H__
#define __EXP_THROUGHPUT_H__

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "experiment.h"
#include "statistics.h"

extern Data * _data;
extern pthread_mutex_t g_write_lock;

class ExperimentThroughput : public Experiment
{ 
public:
  std::chrono::high_resolution_clock::time_point _start_time, _end_time;
  static unsigned long _iops;
  static std::mutex _iops_lock;
  
  ExperimentThroughput(struct ProgramOptions options): Experiment(options) 
  {
    _test_name = "throughput";
  }

  void initialize_custom(unsigned core) override
  {
    _start_time = std::chrono::high_resolution_clock::now();
  }

  bool do_work(unsigned core) override 
  {
    // handle first time setup
    if (_first_iter) 
      {
        PLOG("[%u] Starting Throughput experiment (value len:%lu)...", core, g_data->value_len());
        _first_iter = false;
#if 1
	/* DAX inifialization is serialized by thread (due to libpmempool behavior).
	 * It is in some sense unfair to start measurement in initialize_custom,
	 * which occurs before DAX initialization. Reset start of measurement to first
	 * put operation.
	 */
        _start_time = std::chrono::high_resolution_clock::now();
#endif
      }     

    // end experiment if we've reached the total number of components
    if (_i == _pool_num_objects)
      {
        PINF("[%u] throughput: reached total number of objects. Exiting.", core);
        return false;
      }

    assert(g_data);
    int rc = _store->put(_pool, g_data->key_as_string(_i), g_data->value(_i), g_data->value_len());
    assert(rc == S_OK);

    _i++;
    return true;
  }

  void cleanup_custom(unsigned core)  
  {
    _end_time = std::chrono::high_resolution_clock::now();

    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end_time - _start_time).count() / 1000.0;
    unsigned long iops = (unsigned long) ((double) _pool_num_objects) / secs;
    PLOG("%lu iops (core=%u)", iops, core);
    _iops_lock.lock();
    _iops+= iops;
    _iops_lock.unlock();
  }

  static void summarize() {
    PMAJOR("total IOPS: %lu", _iops);
  }
  
};


#endif //  __EXP_THROUGHPUT_H_
