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
    std::string _outputDirectory = "get_direct_latency";

    ExperimentGetDirectLatency(Component::IKVStore * arg) : Experiment(arg) 
    {
        assert(arg);
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
    }

    void cleanup_custom(unsigned core)  
    {
        boost::filesystem::path dir(_outputDirectory);
        if (boost::filesystem::create_directory(dir))
        {
            std::cout << "Created directory for testing: " << _outputDirectory << std::endl;
        }

        // write one core per file for now. TODO: use synchronization construct for experiments
        std::ostringstream filename;
        filename << _outputDirectory << "/" << core << ".log";

        std::cout << "filename = " << filename.str() << std::endl;

       // output latency info to file 
       std::ofstream outf(filename.str());

       if (!outf)
       {
           std::cerr << "Failed to open file " << _outputDirectory << " for writing" << std::endl;
            exit(1);
       }

       for (int i = 0; i < _pool_num_components; i++)
       {
            outf << _latency[i] << std::endl;
       }
    }
};


#endif //  __EXP_GET_DIRECT_LATENCY_H__
