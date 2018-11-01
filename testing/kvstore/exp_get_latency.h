#ifndef __EXP_GET_LATENCY_H__
#define __EXP_GET_LATENCY_H__

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

class ExperimentGetLatency : public Experiment
{ 
public:
    float _cycles_per_second;  // initialized in do_work first run
    std::vector<double> _start_time;
    std::vector<double> _latencies;
    std::chrono::high_resolution_clock::time_point _exp_start_time;
    BinStatistics _latency_stats;

    ExperimentGetLatency(struct ProgramOptions options): Experiment(options)
    {
        _test_name = "get_latency";

        if (!options.store)
        {
            perror("ExperimentGetLatency passed invalid store");
        }
    }

    void initialize_custom(unsigned core)
    {
        _cycles_per_second = Core::get_rdtsc_frequency_mhz() * 1000000;
        _start_time.resize(_pool_num_components);
        _latencies.resize(_pool_num_components);

        // seed the pool with elements from _data
        _populate_pool_to_capacity(core);
        
        PLOG("pool seeded with values\n");

        _latency_stats.init(_bin_count, _bin_threshold_min, _bin_threshold_max);
    }

    void do_work(unsigned core) override 
    {
        // handle first time setup
        if(_first_iter) 
        {
            PLOG("Starting Get Latency experiment...");

            _first_iter = false;
            _exp_start_time = std::chrono::high_resolution_clock::now();
        }     

        // end experiment if we've reached the total number of components
        if (_i == _pool_num_components)
        {
            PINF("reached total number of components. Exiting.");
            throw std::exception();
        }

        // check time it takes to complete a single put operation
        unsigned int cycles, start, end;
        void * pval;
        size_t pval_len;
        int rc;

        timer.start();
        start = rdtsc();
        rc = _store->get(_pool, _data->key(_i), pval, pval_len);
        end = rdtsc();
        timer.stop();

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
        _start_time.at(_i) = time_since_start;
        _latencies.at(_i) = time;

        if (rc != S_OK)
        {
            std::cout << "rc != S_OK: " << rc << ". Exiting." << std::endl;
            throw std::exception();
        }

        _i++;  // increment after running so all elements get used

       if (_i == _pool_element_end)
       {
            _erase_pool_entries_in_range(_pool_element_start, _pool_element_end);
           _populate_pool_to_capacity(core);

           if (_verbose)
           {
              std::stringstream debug_message;
              debug_message << "pool repopulated: " << _i;
              _debug_print(core, debug_message.str());
           }
       }
    }

    void cleanup_custom(unsigned core)  
    {
        timer.stop();  // just in case; normal code should have already stopped by now

        double run_time = timer.get_time_in_seconds();
        double iops = ((double) _i / run_time);
        PINF("[%u] get: IOPS: %2g in %2g seconds", core, iops, run_time);

       // compute _start_time_stats pre-lock
       BinStatistics start_time_stats = _compute_bin_statistics_from_vectors(_latencies, _start_time, _bin_count, _start_time.front(), _start_time.at(_i-1), _i); 

       pthread_mutex_lock(&g_write_lock);

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
       
       _report_document_save(document, core, experiment_object);
        _print_highest_count_bin(_latency_stats);

       pthread_mutex_unlock(&g_write_lock);
    }
};


#endif //  __EXP_GET_LATENCY_H__
