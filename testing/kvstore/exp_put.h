#ifndef __EXP_PUT_LATENCY_H__
#define __EXP_PUT_LATENCY_H__

#include "experiment.h"

#include "statistics.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

class ExperimentPut : public Experiment
{
  std::size_t _i;
  std::vector<double> _start_time;
  std::vector<double> _latencies;
  BinStatistics _latency_stats;

public:
  ExperimentPut(const ProgramOptions &options)
    : Experiment("put", options)
    , _i(0)
    , _start_time()
    , _latencies()
    , _latency_stats()
  {
  }

  void initialize_custom(unsigned /* core */) override
  {
    _latency_stats = BinStatistics(bin_count(), bin_threshold_min(), bin_threshold_max());
  }

  bool do_work(unsigned core) override
  {
    // handle first time setup
    if ( _first_iter )
    {
      wait_for_delayed_start(core);

      PLOG("[%u] Starting Put experiment...", core);
      _first_iter = false;
    }

    // end experiment if we've reached the total number of components
    if (_i == pool_num_objects())
    {
      PINF("[%u] put: reached total number of components. Exiting.", core);
      return false;
    }

    // check time it takes to complete a single put operation
    try
    {
      StopwatchInterval si(timer);
      auto rc = store()->put(pool(), g_data->key(_i), g_data->value(_i), g_data->value_len());
      if (rc != S_OK)
      {
         auto e = "put returned !S_OK value rc = " + std::to_string(rc);
         PERR("%s.", e.c_str());
         throw std::runtime_error(e);
      }
    }
    catch(...)
    {
      PERR("%s", "put call threw exception! Ending experiment.");
      throw;
    }

    double lap_time = timer.get_lap_time_in_seconds();
    double time_since_start = timer.get_time_in_seconds();

    _update_data_process_amount(core, _i);
    // store the information for later use
    _start_time.push_back(time_since_start);
    _latencies.push_back(lap_time);
    _latency_stats.update(lap_time);

    // THIS IS SKEWING THINGS?
    //_enforce_maximum_pool_size(core, _i);

    ++_i;  // increment after running so all elements get used


    return true;
  }

  void cleanup_custom(unsigned core)
  {
    auto total_time = timer.get_time_in_seconds();
    PLOG("stopwatch : %g secs", total_time);

    {
      unsigned long iops = static_cast<unsigned long>( double(pool_num_objects()) / total_time );
      PLOG("%lu iops (core=%u)", iops, core);
    }

    try
    {
      _debug_print(core, "cleanup_custom started");

      if (is_verbose())
      {
        std::stringstream stats_info;
        stats_info << "creating time_stats with " << bin_count() << " bins: [" << _start_time.front() << " - " << _start_time.at(_i-1) << "]. _i = " << _i << std::endl;
        _debug_print(core, stats_info.str());
      }

      // compute _start_time_stats pre-lock
      BinStatistics start_time_stats = _compute_bin_statistics_from_vectors(_latencies, _start_time, bin_count(), _start_time.front(), _start_time.at(_i-1), _i);
      _debug_print(core, "time_stats created");

      double run_time = timer.get_time_in_seconds();
      unsigned iops = static_cast<unsigned>(double(_i) / run_time);
      PINF("[%u] put: IOPS--> %u (%lu operations over %2g seconds)", core, iops, _i, run_time);
      _update_aggregate_iops(iops);

      double throughput = _calculate_current_throughput();
      PINF("[%u] put: THROUGHPUT: %.2f MB/s (%lu bytes over %.3f seconds)", core, throughput, total_data_processed(), run_time);

      if ( is_json_reporting() )
      {
        std::lock_guard<std::mutex> g(g_write_lock);
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
      }
    }
    catch(...)
    {
      PERR("%s", "cleanup_custom failed inside exp_put.h");
      throw;
    }
  }
};


#endif //  __EXP_PUT_LATENCY_H__
