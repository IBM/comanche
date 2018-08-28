#ifndef __EXP_GET_DIRECT_LATENCY_H__
#define __EXP_GET_DIRECT_LATENCY_H__

#include <core/physical_memory.h> 
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "common/cycles.h"
#include "experiment.h"
#include "kvstore_perf.h"

extern Data * _data;

class ExperimentGetDirectLatency : public Experiment
{ 
public:
    float _cycles_per_second;  // initialized in do_work first run
    std::vector<double> _latency;

    ExperimentGetDirectLatency(struct ProgramOptions options) : Experiment(options) 
    {    
        _test_name = "get_direct_latency";
        
        assert(options.store);
    }

    void initialize_custom(unsigned core)
    {
        _cycles_per_second = Core::get_rdtsc_frequency_mhz() * 1000000;
        _latency.resize(_pool_num_components);

        // seed the pool with elements from _data
        int rc;
        for (int i = 0; i < _pool_num_components; i++)
        {
            rc = _store->put(_pool, _data->key(i), _data->value(i), _data->value_len());
            assert(rc == S_OK);
        }
        PLOG("pool seeded with values\n");
    }

    void do_work(unsigned core) override 
    {
        // handle first time setup
        if(_first_iter) 
        {
            PLOG("Starting Put Latency experiment...");

            _first_iter = false;
        }     

        // end experiment if we've reached the total number of components
        if (_i == _pool_num_components)
        {
            throw std::exception();
        }

        // check time it takes to complete a single put operation
        unsigned int cycles, start, end;

        io_buffer_t handle;
        Core::Physical_memory mem_alloc;
        void * pval = nullptr;
        size_t pval_len = 64;

        if (_component.compare("nvmestore") == 0)
        {
            // TODO: make the input parameters 1 and 2 variable based on experiment inputs
            handle = mem_alloc.allocate_io_buffer(MB(8), 4096, Component::NUMA_NODE_ANY);  
            assert(handle);  
            pval = mem_alloc.virt_addr(handle);
        }
 
        start = rdtsc();
        int rc = _store->get_direct(_pool, _data->key(_i), pval, pval_len);
        end = rdtsc();

        cycles = end - start;
        double time = (cycles / _cycles_per_second);
        //printf("start: %u  end: %u  cycles: %u seconds: %f\n", start, end, cycles, time);
       
        if (_component.compare("nvmestore") == 0)
        { 
             mem_alloc.free_io_buffer(handle);
        }
        else
        {
            free(pval);
        }

        // store the information for later use
        _latency.at(_i) = time;
        assert(rc == S_OK);

        _i++;  // increment after running so all elements get used

        _enforce_maximum_pool_size(core);
    }

    void cleanup_custom(unsigned core)  
    {
        pthread_mutex_lock(&g_write_lock);

       // get existing results, read to document variable
       rapidjson::Document document = _get_report_document();

       // add per-core results here
       rapidjson::Value temp_array(rapidjson::kArrayType);
 
       for (int i = 0; i < _pool_num_components; i++)  
       {
            temp_array.PushBack(_latency[i], document.GetAllocator());
       }

       // add new info to report
       _report_document_save(document, core, temp_array);

       pthread_mutex_unlock(&g_write_lock);
    }
};


#endif //  __EXP_GET_DIRECT_LATENCY_H__
