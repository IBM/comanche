#ifndef __EXP_PUT_LATENCY_H__
#define __EXP_PUT_LATENCY_H__

#include "experiment.h"
#include "kvstore_perf.h"

extern Data * _data;

class ExperimentPutLatency : public Experiment
{ 
public:
 
    ExperimentPutLatency(Component::IKVStore * arg) : Experiment(arg) 
    {
        assert(arg);
    }

    void do_work(unsigned core) override 
    {
        if(_first_iter) 
        {
            PLOG("Starting Put Latency experiment...");

            _first_iter = false;
        }     

        _i++;

        if (_i == _pool_num_components) throw std::exception(); // end experiment

        int rc = _store->put(_pool, _data->key(_i), _data->value(_i), _data->value_len());

        assert(rc == S_OK);
    }

    void cleanup_custom(unsigned core)  
    {
        _end = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / 1000.0;
        PINF("*Put* (%u) IOPS: %2g", core, ((double) _i) / secs); 
    }
};


#endif //  __EXP_PUT_LATENCY_H__
