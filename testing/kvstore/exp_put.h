#ifndef __EXP_PUT_H__
#define __EXP_PUT_H__

#include "experiment.h"

extern Data * _data;

class ExperimentPut : public Experiment
{ 
public:
  
    ExperimentPut(Component::IKVStore * arg) : Experiment(arg) 
    {
        assert(arg);
    }
  
    void do_work(unsigned core) override 
    {
        if(_first_iter) 
        {
            PLOG("Starting Put experiment...");

            _start = std::chrono::high_resolution_clock::now();
            _first_iter = false;
        }
      
        _i++;

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


#endif //  __EXP_PUT_H__
