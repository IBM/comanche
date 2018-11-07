#ifndef __EXP_GET_DIRECT_LATENCY_H__
#define __EXP_GET_DIRECT_LATENCY_H__

#include <chrono>
#include <core/physical_memory.h> 
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/mman.h>

#include "common/cycles.h"
#include "experiment.h"
#include "kvstore_perf.h"
#include "statistics.h"

extern Data * _data;

class ExperimentGetDirect: public Experiment
{ 
public:
    float _cycles_per_second;  // initialized in do_work first run
    std::vector<double> _start_time;
    std::vector<double> _latencies;
    std::chrono::high_resolution_clock::time_point _exp_start_time;
    BinStatistics _latency_stats;
    Component::IKVStore::memory_handle_t _direct_memory_handle = Component::IKVStore::HANDLE_NONE;

    ExperimentGetDirect(struct ProgramOptions options) : Experiment(options) 
    {    
        _test_name = "get_direct";
        
        if (!options.store)
        {
            perror("ExperimentGetDirect passed an invalid store");
        }
    }

    void initialize_custom(unsigned core)
    {
        _cycles_per_second = Core::get_rdtsc_frequency_mhz() * 1000000;
        _start_time.resize(_pool_num_components);
        _latencies.resize(_pool_num_components);

        if (_component.compare("dawn_client") == 0)
        {
           size_t data_size = sizeof(KV_pair) * _data->_num_elements;
           Data * data = (Data*)aligned_alloc(_pool_size, data_size);
           madvise(data, data_size, MADV_HUGEPAGE);
           _direct_memory_handle = _store->register_direct_memory(data, data_size);
        }

        // seed the pool with elements from _data
        _populate_pool_to_capacity(core, _direct_memory_handle);

        PLOG("pool seeded with values\n");

        _latency_stats.init(_bin_count, _bin_threshold_min, _bin_threshold_max);
    }

    void do_work(unsigned core) override 
    {
        // handle first time setup
        if(_first_iter) 
        {
            PLOG("Starting Get Direct experiment...");

            _first_iter = false;
            _exp_start_time = std::chrono::high_resolution_clock::now();
        }     

        // end experiment if we've reached the total number of components
        if (_i == _pool_num_components)
        {
            timer.stop();
            throw std::exception();
        }

        // check time it takes to complete a single put operation
        unsigned int cycles, start, end;

        io_buffer_t handle;
        Core::Physical_memory mem_alloc;
        size_t pval_len = 64;
        void * pval = malloc(pval_len);
        Component::IKVStore::memory_handle_t memory_handle = Component::IKVStore::HANDLE_NONE; 

        if (_component.compare("nvmestore") == 0)
        {
            // TODO: make the input parameters 1 and 2 variable based on experiment inputs
            handle = mem_alloc.allocate_io_buffer(MB(8), 4096, Component::NUMA_NODE_ANY);  
            
            if (!handle)
            {
                perror("ExpGetDirect.do_work: allocate_io_buffer failed");
            }

            pval = mem_alloc.virt_addr(handle);
        }
        else if (_component.compare("dawn_client") == 0)
        {
            memory_handle = _direct_memory_handle;
        }
 
        timer.start();
        start = rdtsc();
        int rc = _store->get_direct(_pool, _data->key(_i), pval, pval_len, memory_handle);
        end = rdtsc();
        timer.stop();
        
        cycles = end - start;
        double time = (cycles / _cycles_per_second);
        //printf("start: %u  end: %u  cycles: %u seconds: %f\n", start, end, cycles, time);

        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        double time_since_start = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - _exp_start_time).count() / 1000.0;
       
        if (_component.compare("nvmestore") == 0)
        { 
             mem_alloc.free_io_buffer(handle);
        }

        if (pval != nullptr)
        {
            free(pval);
        }

        // store the information for later use
        _latencies.at(_i) = time;
        _start_time.at(_i) = time_since_start; 
        _latency_stats.update(time);
        
        if (rc != S_OK)
        {
            std::cout << "rc != S_OK: rc = " << rc << std::endl;
            throw std::exception();
        }

        _i++;  // increment after running so all elements get used

       if (_i == _pool_element_end)
       {
            _erase_pool_entries_in_range(_pool_element_start, _pool_element_end);
           _populate_pool_to_capacity(core, _direct_memory_handle);

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
        if (_component.compare("dawn_client") == 0)
        {
            _store->unregister_direct_memory(_direct_memory_handle);
        }

        double run_time = timer.get_time_in_seconds();
        double iops = _i / run_time;
        PINF("[%u] get_direct: IOPS: %2g in %2g seconds", core, iops, run_time);

        if (_verbose)
        {
            std::stringstream stats_info;
            stats_info << "creating time_stats with " << _bin_count << " bins: [" << _start_time.front() << " - " << _start_time.at(_i-1) << "]" << std::endl;
            _debug_print(core, stats_info.str());
        }

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
        _print_highest_count_bin(_latency_stats);

       _report_document_save(document, core, experiment_object);

       pthread_mutex_unlock(&g_write_lock);

    }
};


#endif //  __EXP_GET_DIRECT_LATENCY_H__
