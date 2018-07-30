#ifndef __EXP_PUT_LATENCY_H__
#define __EXP_PUT_LATENCY_H__

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "common/cycles.h"
#include "experiment.h"
#include "kvstore_perf.h"

extern Data * _data;

class ExperimentPutLatency : public Experiment
{ 
public:
    float _cycles_per_second;  // initialized in do_work first run
    std::vector<double> _latency;
    std::string outputFilename = "put_latency.log";

    ExperimentPutLatency(Component::IKVStore * arg) : Experiment(arg) 
    {
        assert(arg);
    }

    void initialize_custom(unsigned core)
    {
        _cycles_per_second = Core::get_rdtsc_frequency_mhz() * 1000000;
        _latency.resize(_pool_num_components);
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

        start = rdtsc();
        int rc = _store->put(_pool, _data->key(_i), _data->value(_i), _data->value_len());
        end = rdtsc();
        cycles = end - start;
        double time = (cycles / _cycles_per_second);
        //printf("start: %u  end: %u  cycles: %u seconds: %f\n", start, end, cycles, time);

        // store the information for later use
        _latency.at(_i) = time;

        assert(rc == S_OK);

        _i++;  // increment after running so all elements get used
    }

    void cleanup_custom(unsigned core)  
    {
       // output latency info to file 
       std::ofstream outf(outputFilename);

       if (!outf)
       {
           std::cerr << "Failed to open file " << outputFilename << " for writing" << std::endl;
            exit(1);
       }

       for (int i = 0; i < _pool_num_components; i++)
       {
            outf << _latency[i] << std::endl;
       }
    }
};


#endif //  __EXP_PUT_LATENCY_H__
