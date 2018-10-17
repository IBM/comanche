#ifndef __EXP_GET_H__
#define __EXP_GET_H__

#include "data.h"
#include "experiment.h"

extern Data * _data;
extern pthread_mutex_t g_write_lock;

class ExperimentGet : public Experiment
{ 
public:

    ExperimentGet(struct ProgramOptions options) : Experiment(options) 
    {
        _test_name = "get";
    }
    
    void initialize_custom(unsigned core)
    { 
        PLOG("(%u) Populating key/value pairs for Get test...", core);

        _populate_pool_to_capacity(core);

        PLOG("(%u) KVPs populated.", core);
    }

    void do_work(unsigned core) override 
    {
        if(_first_iter) 
        {
            PLOG("Starting Get experiment...");

            _start = std::chrono::high_resolution_clock::now();
            timer.start();

            _first_iter = false;
        }
  
        if(_i == _data->num_elements()) 
        { 
            timer.stop();
            throw std::exception();
        }
  
        void * pval;
        size_t pval_len;
        
        int rc = _store->get(_pool, _data->key(_i), pval, pval_len);

        assert(rc == S_OK);

        free(pval);
        _i++;

       if (_i == _pool_element_end)
       {
           timer.stop();
            _erase_pool_entries_in_range(_pool_element_start, _pool_element_end);
           _populate_pool_to_capacity(core); 
           timer.start();
       }
    }
    
    void cleanup_custom(unsigned core) 
    { 
        timer.stop();  // just in case; normal code should have already stopped by now

        double run_time = timer.get_time_in_seconds();
        double iops = ((double) _i / run_time);
        PINF("[get] Timer: IOPS: %2g in %2g", iops, run_time);

       pthread_mutex_lock(&g_write_lock);

       // get existing results, read to document variable
       rapidjson::Document document = _get_report_document();

       // add per-core results here
       rapidjson::Value temp_value;
       temp_value.SetDouble(iops);

       // add new info to report
       _report_document_save(document, core, temp_value);

       pthread_mutex_unlock(&g_write_lock);
    }
};

#endif // __EXP_GET_H__
