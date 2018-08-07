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
    std::string _test_name = "get_latency";
    std::string _report_filename;

    ExperimentGetLatency(struct ProgramOptions options): Experiment(options.store)
    {
       _report_filename = options.report_file_name;

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
        void * pval;
        size_t pval_len;

        start = rdtsc();
        int rc = _store->get(_pool, _data->key(_i), pval, pval_len);
        end = rdtsc();

        cycles = end - start;
        double time = (cycles / _cycles_per_second);
        //printf("start: %u  end: %u  cycles: %u seconds: %f\n", start, end, cycles, time);
        
        free(pval);

        // store the information for later use
        _latency.at(_i) = time;

        assert(rc == S_OK);

        _i++;  // increment after running so all elements get used
    }

    void cleanup_custom(unsigned core)  
    {
       // get existing results, read to document variable
       pthread_mutex_lock(&g_write_lock);

       FILE *pFile = fopen(_report_filename.c_str(), "r+");
       rapidjson::FileStream is(pFile);
       rapidjson::Document document;
       document.ParseStream<0>(is);

       rapidjson::Value temp_value;
       rapidjson::Value temp_object(rapidjson::kObjectType);
       rapidjson::Value temp_array(rapidjson::kArrayType);

       // add per-core results here        
       for (int i = 0; i < _pool_num_components; i++)  
       {
            temp_array.PushBack(_latency[i], document.GetAllocator());
       }

       std::string core_string = std::to_string(core);
       temp_value.SetString(rapidjson::StringRef(core_string.c_str()));

       if (!document.HasMember(_test_name.c_str()))
       {
           temp_object.AddMember(temp_value, temp_array, document.GetAllocator());
            document.AddMember(rapidjson::StringRef(_test_name.c_str()), temp_object, document.GetAllocator()); 
       }
       else 
       {
            rapidjson::Value &items = document[_test_name.c_str()];
            &items.AddMember(temp_value, temp_array, document.GetAllocator());
       }

        // write back to file
       rapidjson::StringBuffer strbuf;
       rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
       document.Accept(writer);

        std::ofstream outf(_report_filename.c_str());
        outf << strbuf.GetString() << std::endl; 

       pthread_mutex_unlock(&g_write_lock);
    }
};


#endif //  __EXP_GET_LATENCY_H__
