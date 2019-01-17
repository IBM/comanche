#ifndef __EXP_GET_LATENCY_H__
#define __EXP_GET_LATENCY_H__

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

class ExperimentGet : public Experiment
{ 
public:
  std::vector<double> _start_time;
  std::vector<double> _latencies;
  std::chrono::high_resolution_clock::time_point _exp_start_time;
  BinStatistics _latency_stats;

  ExperimentGet(struct ProgramOptions options): Experiment(options)
  {
    _test_name = "get";
  }

  void initialize_custom(unsigned core) override
  {
    _latency_stats.init(_bin_count, _bin_threshold_min, _bin_threshold_max);
     
    PLOG("[%u] exp_get: pool seeded with values", core);
  }

  bool do_work(unsigned core) override
  {
    // handle first time setup
    if(_first_iter) 
      {
        _pool_element_end = -1;
        // seed the pool with elements from _data
        _populate_pool_to_capacity(core);

        PLOG("[%u] Starting Get experiment...", core);

        _first_iter = false;
        _exp_start_time = std::chrono::high_resolution_clock::now();
      }     

    // end experiment if we've reached the total number of components
    if (_i + 1 == _pool_num_objects)
      {
        PINF("[%u] reached total number of components. Exiting.", core);
        return false; 
      }

    // check time it takes to complete a single put operation
    uint64_t cycles, start, end;
    void * pval;
    size_t pval_len;
    int rc;

    timer.start();
    start = rdtsc();

    try
      {
        rc = _store->get(_pool, g_data->key(_i), pval, pval_len);
      }
    catch(...)
      {
        PERR("get call failed! Ending experiment.");
        throw std::exception();
      }
    end = rdtsc();
    timer.stop();

    _update_data_process_amount(core, _i);

    cycles = end - start;
    double time = (cycles / _cycles_per_second);
    //printf("start: %u  end: %u  cycles: %u seconds: %f\n", start, end, cycles, time);

    std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
    double time_since_start = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - _exp_start_time).count() / 1000.0;

    if (pval != nullptr)
      {
        _store->free_memory(pval);
      }

    // store the information for later use
    _latency_stats.update(time);
    _start_time.push_back(time_since_start);
    _latencies.push_back(time);

    if (rc != S_OK)
      {
        std::cerr << "_pool_element_end = " << _pool_element_end << std::endl;
        std::cerr << "rc != S_OK: " << rc << " @ _i = " << _i << ". Exiting." << std::endl;
        throw std::exception();
      }

    _i++;  // increment after running so all elements get used

    if (_i == _pool_element_end + 1)
      {
        try
          {
            _erase_pool_entries_in_range(_pool_element_start, _pool_element_end);
            _populate_pool_to_capacity(core);
          }
        catch(...)
          {
            PERR("failed during erasing and repopulation");
            throw std::exception();
          }

        if (_verbose)
          {
            std::stringstream debug_message;
            debug_message << "pool repopulated: " << _i;
            _debug_print(core, debug_message.str());
          }
      }
    return true;
  }

  void cleanup_custom(unsigned core)  
  {
    timer.stop();  // just in case; normal code should have already stopped by now

    double run_time = timer.get_time_in_seconds();
    double iops = ((double) _i / run_time);
    PINF("[%u] get: IOPS: %2g in %2g seconds", core, iops, run_time);
    _update_aggregate_iops(iops);
    
    double throughput = _calculate_current_throughput();
    PINF("[%u] get: THROUGHPUT: %.2f MB/s (%ld bytes over %.3f seconds)", core, throughput, _total_data_processed, run_time);

    // compute _start_time_stats pre-lock
    BinStatistics start_time_stats = _compute_bin_statistics_from_vectors(_latencies, _start_time, _bin_count, _start_time.front(), _start_time.at(_i-1), _i); 

    if (_skip_json_reporting)
    {
      return;
    }

    pthread_mutex_lock(&g_write_lock);

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
    _print_highest_count_bin(_latency_stats, core);

    pthread_mutex_unlock(&g_write_lock);
  }
};


#endif //  __EXP_GET_LATENCY_H__
