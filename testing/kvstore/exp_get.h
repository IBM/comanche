#ifndef __EXP_GET_H__
#define __EXP_GET_H__

#include "data.h"
#include "experiment.h"

extern Data * _data;


class ExperimentGet : public Experiment
{ 
public:

    ExperimentGet(Component::IKVStore * arg) : Experiment(arg) 
    {
        assert(arg);
    }
    
    void initialize_custom(unsigned core)
    { 
        PLOG("(%u) Populating key/value pairs for Get test...", core);

        for(size_t i=0;i<_data->num_elements();i++) 
        {
            int rc = _store->put(_pool, _data->key(i), _data->value(i), _data->value_len());

            if(rc != S_OK) 
            {
                PERR("store->put return code: %d", rc);
            }
        }

        PLOG("(%u) KVPs populated.", core);
    }

    void do_work(unsigned core) override 
    {
        if(_first_iter) 
        {
            PLOG("Starting Get experiment...");
            _start = std::chrono::high_resolution_clock::now();
            _first_iter = false;
        }
  
        if(_i == _data->num_elements()) 
        { 
            throw std::exception();
        }
  
        void * pval;
        size_t pval_len;
        
        int rc = _store->get(_pool, _data->key(_i), pval, pval_len);

        assert(rc == S_OK);

        free(pval);
        _i++;
    }
    
    void cleanup_custom(unsigned core) 
    { 
        _end = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / 1000.0;
        PINF("*Get* (%u) IOPS: %lu", core, (uint64_t) (((double) _i) / secs));
    }
};


#endif // __EXP_GET_H__
