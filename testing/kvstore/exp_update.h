#ifndef __EXP_UPDATE_H__
#define __EXP_UPDATE_H__

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "experiment.h"
#include "kvstore_perf.h"
#include "statistics.h"

extern Data * g_data;
extern pthread_mutex_t g_write_lock;

class ExperimentUpdate : public Experiment
{ 
public:
  std::vector<double> _start_time;
  std::vector<double> _latencies;
  BinStatistics _latency_stats;

  ExperimentUpdate(struct ProgramOptions options): Experiment(options) 
  {
    _test_name = "update";
  }

  void initialize_custom(unsigned core) override
  {
    _latency_stats.init(_bin_count, _bin_threshold_min, _bin_threshold_max);
  }

  bool do_work(unsigned core) override 
  {
    // handle first time setup
    if(_first_iter) {
      PLOG("[%u] Starting Update experiment...", core);
      _pool_element_end = -1;

      // seed the pool with elements from _data
      _populate_pool_to_capacity(core);

      _first_iter = false;
    }

    // end experiment if we've reached the total number of components
    if (_i == _pool_num_objects) {
      PINF("[%u] update: reached total number of components. Exiting.", core);
      return false;
    }

    // generate a new random value with the same value length to use
    auto new_val = Common::random_string(g_data->value_len());

    // check time it takes to complete a single put operation
    int rc;

    timer.start();

    try {
        rc = _store->put(_pool, g_data->key(_i), new_val.c_str(), g_data->value_len());
    }
    catch(...) {
      PERR("put (update) call failed! Returned %d. Ending experiment.", rc);
      throw std::exception();
    }

    timer.stop();
   
    double lap_time = timer.get_lap_time_in_seconds();
    double time_since_start = timer.get_time_in_seconds();

    _update_data_process_amount(core, _i);

    // store the information for later use
    _start_time.push_back(time_since_start);
    _latencies.push_back(lap_time);
    _latency_stats.update(lap_time);

    _enforce_maximum_pool_size(core);

    _i++;  // increment after running so all elements get used

    if (rc != S_OK) {
      PERR("[%u] returned !S_OK (%d) on put call! Exiting.", core, rc);
      return false;
    }
/* // Temporarily disabled since erase step isn't supported on all components yet -TJanssen 1/30/2019
    if (_i == _pool_element_end + 1) {
      try {
        _erase_pool_entries_in_range(_pool_element_start, _pool_element_end);
        _populate_pool_to_capacity(core);
      }
      catch(...) {
        PERR("failed during erasing and repopulation");
        throw std::exception();
      }

      if (_verbose) {
        std::stringstream debug_message;
        debug_message << "pool repopulated: " << _i;
        _debug_print(core, debug_message.str());
      }
    }
*/
    return true;
  }

  void cleanup_custom(unsigned core)  
  {
    try {
      _debug_print(core, "cleanup_custom started");

      if (_verbose) {
        std::stringstream stats_info;
        stats_info << "creating time_stats with " << _bin_count << " bins: [" << _start_time.front() << " - " << _start_time.at(_i-1) << "]. _i = " << _i << std::endl;
        _debug_print(core, stats_info.str());
      }

      // compute _start_time_stats pre-lock
      BinStatistics start_time_stats = _compute_bin_statistics_from_vectors(_latencies, _start_time, _bin_count, _start_time.front(), _start_time.at(_i-1), _i);
      _debug_print(core, "time_stats created"); 

      double run_time = timer.get_time_in_seconds();
      unsigned iops = ((double)_i) / run_time;
      PINF("[%u] update: IOPS--> %u (%lu operations over %2g seconds)", core, iops, _i, run_time);
      _update_aggregate_iops(iops);

      double throughput = _calculate_current_throughput();
      PINF("[%u] update: THROUGHPUT: %.2f MB/s (%ld bytes over %.3f seconds)", core, throughput, _total_data_processed, run_time);

      if (_skip_json_reporting) {
        return;
      }

      pthread_mutex_lock(&g_write_lock);
      _debug_print(core, "cleanup_custom mutex locked");

      // get existing results, read to document variable
      rapidjson::Document document = _get_report_document();

      // collect latency stats
      rapidjson::Value latency_object = _add_statistics_to_report("latency", _latency_stats, document);
      rapidjson::Value timing_object = _add_statistics_to_report("start_time", start_time_stats, document);
      rapidjson::Value iops_object;
      rapidjson::Value throughput_object;

      iops_object.SetDouble(iops);
      throughput_object.SetDouble(throughput);

      // save everything
      rapidjson::Value experiment_object(rapidjson::kObjectType);
       
      experiment_object.AddMember("IOPS", iops_object, document.GetAllocator());
      experiment_object.AddMember("throughput (MB/s)", throughput_object, document.GetAllocator());
      experiment_object.AddMember("latency", latency_object, document.GetAllocator());
      experiment_object.AddMember("start_time", timing_object, document.GetAllocator()); 
      _print_highest_count_bin(_latency_stats, core);

      _report_document_save(document, core, experiment_object);

      _debug_print(core, "cleanup_custom mutex unlocking");
      pthread_mutex_unlock(&g_write_lock);
    }
    catch(...) {
      PERR("cleanup_custom failed inside exp_put.h");
      throw std::exception();
    }
  }
};


#endif //  __EXP_UPDATE_H__
