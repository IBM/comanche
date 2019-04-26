#ifndef __EXP_PUT_DIRECT_LATENCY_H__
#define __EXP_PUT_DIRECT_LATENCY_H__

#include "experiment.h"

#include "statistics.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

class ExperimentPutDirect : public Experiment
{
  std::size_t _i;
    std::vector<double> _start_time;
    std::vector<double> _latencies;
    std::chrono::high_resolution_clock::time_point _exp_start_time;
    BinStatistics _latency_stats;

public:
    ExperimentPutDirect(const ProgramOptions &options)
      : Experiment("put_direct", options)
      , _i(0)
      , _start_time()
      , _latencies()
      , _exp_start_time()
      , _latency_stats()
    {
    }

    void initialize_custom(unsigned core)
    {
        _latency_stats = BinStatistics(bin_count(), bin_threshold_min(), bin_threshold_max());

        _debug_print(core, "initialize_custom done");
    }

    bool do_work(unsigned core) override
    {
        // handle first time setup
        if(_first_iter)
        {
          wait_for_delayed_start(core);

          PLOG("[%u] Starting Put Direct experiment...", core);

          _first_iter = false;
          _exp_start_time = std::chrono::high_resolution_clock::now();
        }

        // end experiment if we've reached the total number of components
        if ( _i == pool_num_objects() )
        {
            timer.stop();
            PINF("[%u] put_direct: reached total number of components. Exiting.", core);
            return false;
        }

        // check time it takes to complete a single put operation

        {
          StopwatchInterval si(timer);
          auto rc = store()->put_direct(pool(), g_data->key(_i), g_data->value(_i), g_data->value_len(), memory_handle());
          if (rc != S_OK)
          {
            auto e = "put_direct returned !S_OK value rc = " + std::to_string(rc);
            PERR("%s.", e.c_str());
            throw std::runtime_error(e);
          }
        }

        _update_data_process_amount(core, _i);

        auto time = timer.get_lap_time_in_seconds();

        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        double time_since_start = double(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - _exp_start_time).count()) / 1000.0;

        // store the information for later use
        _start_time.push_back(time_since_start);
        _latencies.push_back(time);

        _latency_stats.update(time);

        ++_i;  // increment after running so all elements get used

        _enforce_maximum_pool_size(core, _i);

        return true;
    }

    void cleanup_custom(unsigned core)
    {
        _debug_print(core, "cleanup_custom started");

        timer.stop();  // should already be stopped here; just in case
        double run_time = timer.get_time_in_seconds();
        double iops = double(_i) / run_time;
        PINF("[%u] put_direct: IOPS: %2g in %2g seconds", core, iops, run_time);
        _update_aggregate_iops(iops);

        double throughput = _calculate_current_throughput();
        PINF("[%u] put_direct: THROUGHPUT: %.2f MB/s (%lu bytes over %.3f seconds)", core, throughput, total_data_processed(), run_time);

        if ( is_verbose() )
        {
            std::stringstream stats_info;
            stats_info << "creating time_stats with " << bin_count() << " bins: [" << _start_time.front() << " - " << _start_time.at(_i-1) << "]. _i = " << _i << std::endl;
            _debug_print(core, stats_info.str());
        }

      if ( is_json_reporting() )
      {
         // compute _start_time_stats pre-lock
         BinStatistics start_time_stats = _compute_bin_statistics_from_vectors(_latencies, _start_time, bin_count(), _start_time.front(), _start_time.at(_i-1), _i);
         _debug_print(core, "time_stats created");

        try
        {
          // save everything

          std::lock_guard<std::mutex> g(g_write_lock);
          _debug_print(core, "cleanup_custom mutex locked");
          // get existing results, read to document variable
          rapidjson::Document document = _get_report_document();
          rapidjson::Value experiment_object(rapidjson::kObjectType);
          experiment_object
            .AddMember("IOPS", double(iops), document.GetAllocator())
            .AddMember("throughput (MB/s)", double(throughput), document.GetAllocator())
            .AddMember("latency", _add_statistics_to_report(_latency_stats, document), document.GetAllocator())
            .AddMember("start_time", _add_statistics_to_report(start_time_stats, document), document.GetAllocator())
            ;

          _report_document_save(document, core, experiment_object);
        }
        catch(...)
        {
          PERR("%s", "failed during save to JSON");
          throw;
        }

        _print_highest_count_bin(_latency_stats, core);

        _debug_print(core, "cleanup_custom mutex unlocking");
      }
    }
};


#endif //  __EXP_PUT_DIRECT_LATENCY_H__
