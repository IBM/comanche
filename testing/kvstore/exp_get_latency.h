#ifndef __EXP_GET_LATENCY_H__
#define __EXP_GET_LATENCY_H__

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "common/cycles.h"
#include "experiment.h"
#include "kvstore_perf.h"

extern Data * _data;
extern pthread_mutex_t g_write_lock;

class ExperimentGetLatency : public Experiment
{ 
public:
    float _cycles_per_second;  // initialized in do_work first run
    std::vector<double> _latency;
    std::vector<double> _start_time;
    unsigned int _start_rdtsc;

    ExperimentGetLatency(struct ProgramOptions options): Experiment(options)
    {
        _test_name = "get_latency";

        assert(options.store); 
    }

    void initialize_custom(unsigned core)
    {
        _cycles_per_second = Core::get_rdtsc_frequency_mhz() * 1000000;
        _latency.resize(_pool_num_components);
        _start_time.resize(_pool_num_components);

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
            _start_rdtsc = rdtsc();
        }     

        // end experiment if we've reached the total number of components
        if (_i == _pool_num_components)
        {
            throw std::exception();
        }

        // check time it takes to complete a single put operation
        unsigned int cycles, start, end;
        void * pval;
        size_t pval_len;
        int rc;

        start = rdtsc();
        rc = _store->get(_pool, _data->key(_i), pval, pval_len);
        end = rdtsc();

        cycles = end - start;
        double time = (cycles / _cycles_per_second);
        //printf("start: %u  end: %u  cycles: %u seconds: %f\n", start, end, cycles, time);
        
        unsigned int cycles_since_start = end - _start_rdtsc;
        double time_since_start = (cycles_since_start / _cycles_per_second);

        free(pval);

        // store the information for later use
        _latency.at(_i) = time;
        _start_time.at(_i) = time_since_start;

        assert(rc == S_OK);

        _i++;  // increment after running so all elements get used
    }

    void cleanup_custom(unsigned core)  
    {
       pthread_mutex_lock(&g_write_lock);

       // get existing results, read to document variable
       rapidjson::Document document = _get_report_document();

       // add per-core results here
       rapidjson::Value temp_array(rapidjson::kArrayType);
       rapidjson::Value temp_value;
 
       for (int i = 0; i < _pool_num_components; i++)  
       {
            // PushBack requires unique object
            rapidjson::Value temp_object(rapidjson::kObjectType); 

            temp_value.SetDouble(_start_time[i]);
            temp_object.AddMember("time_since_start", temp_value, document.GetAllocator());

            temp_value.SetDouble(_latency[i]);
            temp_object.AddMember("latency", temp_value, document.GetAllocator());

            temp_array.PushBack(temp_object, document.GetAllocator());
       }

       // add new info to report
       _report_document_save(document, core, temp_array);

       pthread_mutex_unlock(&g_write_lock);
    }
};


#endif //  __EXP_GET_LATENCY_H__
