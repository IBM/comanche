#ifndef __EXP_PUT_LATENCY_H__
#define __EXP_PUT_LATENCY_H__

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "common/cycles.h"
#include "experiment.h"
#include "kvstore_perf.h"
#include "statistics.h"

extern Data * _data;
extern pthread_mutex_t g_write_lock;

class ExperimentPut : public Experiment
{ 
public:
    float _cycles_per_second;  // initialized in do_work first run
    std::vector<double> _start_time;
    std::vector<double> _latencies;
    std::chrono::high_resolution_clock::time_point _exp_start_time;
    BinStatistics _latency_stats;

    ExperimentPut(struct ProgramOptions options): Experiment(options) 
    {
        _test_name = "put";

        if (!options.store)
        {
            perror("ExperimentPut passed an invalid store");
            throw std::exception();
        }
    }

    void initialize_custom(unsigned core) override
    {
        _latency_stats.init(_bin_count, _bin_threshold_min, _bin_threshold_max);

        _cycles_per_second = Core::get_rdtsc_frequency_mhz() * 1000000;
    }

    void do_work(unsigned core) override 
    {
        // handle first time setup
        if(_first_iter) 
        {
            PLOG("Starting Put experiment...");
            _first_iter = false;
            unsigned int _start_rdtsc = rdtsc(); 
            _exp_start_time = std::chrono::high_resolution_clock::now();
        }     

        // end experiment if we've reached the total number of components
        if (_i == _pool_num_components)
        {
            timer.stop();
            std::cerr << "reached last element. Last _start_time = " << _start_time.at(_i) << std::endl;
            throw std::exception();
        }

        // check time it takes to complete a single put operation
        unsigned int cycles, start, end;
        int rc;

        timer.start();
        start = rdtsc();
        try
        {
          rc = _store->put(_pool, _data->key(_i), _data->value(_i), _data->value_len());
        }
        catch(...)
        {
          PERR("put call failed! Ending experiment.");
          throw std::exception();
        }
        end = rdtsc();
        timer.stop();

        cycles = end - start;
        double time = (cycles / _cycles_per_second);
        //printf("start: %u  end: %u  cycles: %u seconds: %f\n", start, end, cycles, time);

        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        double time_since_start = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - _exp_start_time).count() / 1000.0;

        // store the information for later use
        _start_time.push_back(time_since_start);
        _latencies.push_back(time);

        _latency_stats.update(time);
       
        _i++;  // increment after running so all elements get used

        _enforce_maximum_pool_size(core);

        if (rc != S_OK)
        {
            timer.stop();
            perror("put returned !S_OK value");
            throw std::exception();
        }
    }

    void cleanup_custom(unsigned core)  
    {
        try
        {
          timer.stop();
          _debug_print(core, "cleanup_custom started");

          if (_verbose)
          {
              std::stringstream stats_info;
              stats_info << "creating time_stats with " << _bin_count << " bins: [" << _start_time.front() << " - " << _start_time.at(_i-1) << "]. _i = " << _i << std::endl;
              _debug_print(core, stats_info.str());
          }

         // compute _start_time_stats pre-lock
         BinStatistics start_time_stats = _compute_bin_statistics_from_vectors(_latencies, _start_time, _bin_count, _start_time.front(), _start_time.at(_i-1), _i);
         _debug_print(core, "time_stats created"); 

          double run_time = timer.get_time_in_seconds();
          double iops = _i / run_time;
          PINF("[%u] put: IOPS: %2g in %2g seconds", core, iops, run_time);

          pthread_mutex_lock(&g_write_lock);
         _debug_print(core, "cleanup_custom mutex locked");

         // get existing results, read to document variable
         rapidjson::Document document = _get_report_document();

         // collect latency stats
         rapidjson::Value latency_object = _add_statistics_to_report("latency", _latency_stats, document);
         rapidjson::Value timing_object = _add_statistics_to_report("start_time", start_time_stats, document);
         rapidjson::Value iops_object; 
         iops_object.SetDouble(iops);

         // save everything
         rapidjson::Value experiment_object(rapidjson::kObjectType);
         
         experiment_object.AddMember("IOPS", iops_object, document.GetAllocator());
         experiment_object.AddMember("latency", latency_object, document.GetAllocator());
         experiment_object.AddMember("start_time", timing_object, document.GetAllocator()); 
           _print_highest_count_bin(_latency_stats);
        
         _report_document_save(document, core, experiment_object);

         _debug_print(core, "cleanup_custom mutex unlocking");
         pthread_mutex_unlock(&g_write_lock);
       }
       catch(...)
       {
        PERR("cleanup_custom failed inside exp_put.h");
        throw std::exception();
       }
    }
};


#endif //  __EXP_PUT_LATENCY_H__
