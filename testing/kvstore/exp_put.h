#ifndef __EXP_PUT_H__
#define __EXP_PUT_H__

#include "experiment.h"

extern Data * _data;
extern pthread_mutex_t g_write_lock;

class ExperimentPut : public Experiment
{ 
public:
  
    ExperimentPut(struct ProgramOptions options) : Experiment(options) 
    {
        _test_name = "put";
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
        double iops = ((double) _i) / secs;
        PINF("*Put* (%u) IOPS: %2g", core, iops); 

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


#endif //  __EXP_PUT_H__
