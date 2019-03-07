#ifndef __EXP_PUT_DIRECT_LATENCY_H__
#define __EXP_PUT_DIRECT_LATENCY_H__

#include "experiment.h"
#include "kvstore_perf.h"
#include "statistics.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

extern Data * g_data;
extern pthread_mutex_t g_write_lock;

class ExperimentPutDirect : public Experiment
{ 
public:
    std::vector<double> _start_time;
    std::vector<double> _latencies;
    std::chrono::high_resolution_clock::time_point _exp_start_time;
    BinStatistics _latency_stats;

    ExperimentPutDirect(struct ProgramOptions options): Experiment(options) 
      , _start_time()
      , _latencies()
      , _exp_start_time()
      , _latency_stats()
    {
        _test_name = "put_direct";
    }

    void initialize_custom(unsigned core)
    {
        _latency_stats.init(_bin_count, _bin_threshold_min, _bin_threshold_max);
        
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
        if (_i == _pool_num_objects)
        {
            timer.stop();
            PINF("[%u] put_direct: reached total number of components. Exiting.", core);
            return false; 
        }

        // check time it takes to complete a single put operation
        uint64_t cycles, start, end;
        int rc;

        timer.start();
        start = rdtsc();
        rc = _store->put_direct(_pool, g_data->key(_i), g_data->value(_i), g_data->value_len(), _memory_handle);
        end = rdtsc();
        timer.stop();

        _update_data_process_amount(core, _i);

        cycles = end - start;
        double time = double(cycles) / _cycles_per_second;
        //printf("start: %u  end: %u  cycles: %u seconds: %f\n", start, end, cycles, time);

        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        double time_since_start = double(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - _exp_start_time).count()) / 1000.0;

        // store the information for later use
        _start_time.push_back(time_since_start);
        _latencies.push_back(time);

        _latency_stats.update(time);
       
        _i++;  // increment after running so all elements get used

        _enforce_maximum_pool_size(core);

        if (rc != S_OK)
        {
            perror("put returned !S_OK value");
            std::cout << "rc = " << rc << std::endl;
            throw std::exception();
        }
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
        PINF("[%u] put_direct: THROUGHPUT: %.2f MB/s (%ld bytes over %.3f seconds)", core, throughput, _total_data_processed, run_time);

        if (_verbose)
        {
            std::stringstream stats_info;
            stats_info << "creating time_stats with " << _bin_count << " bins: [" << _start_time.front() << " - " << _start_time.at(_i-1) << "]. _i = " << _i << std::endl;
            _debug_print(core, stats_info.str());
        }

       // compute _start_time_stats pre-lock
       BinStatistics start_time_stats = _compute_bin_statistics_from_vectors(_latencies, _start_time, _bin_count, _start_time.front(), _start_time.at(_i-1), _i);
       _debug_print(core, "time_stats created"); 

      if (_skip_json_reporting)
      {
        return;
      }

       pthread_mutex_lock(&g_write_lock);
       _debug_print(core, "cleanup_custom mutex locked");

       try
       {
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
         
         _report_document_save(document, core, experiment_object);
       }
       catch(...)
       {
         PERR("%s", "failed during save to JSON");
         pthread_mutex_unlock(&g_write_lock);
         throw std::exception();
       }

        _print_highest_count_bin(_latency_stats, core);

       _debug_print(core, "cleanup_custom mutex unlocking");
       pthread_mutex_unlock(&g_write_lock);
    }
};


#endif //  __EXP_PUT_DIRECT_LATENCY_H__
