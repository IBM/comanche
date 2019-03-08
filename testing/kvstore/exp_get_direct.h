#ifndef __EXP_GET_DIRECT_LATENCY_H__
#define __EXP_GET_DIRECT_LATENCY_H__

#include "experiment.h"

#include "statistics.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#include <core/physical_memory.h> 
#pragma GCC diagnostic pop

#include <sys/mman.h>

#include <chrono>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

extern std::mutex g_write_lock;
extern Data * g_data;

class ExperimentGetDirect
  : public Experiment
{ 
  std::size_t _i;
    std::vector<double> _start_time;
    std::vector<double> _latencies;
    std::chrono::high_resolution_clock::time_point _exp_start_time;
    BinStatistics _latency_stats;
    Component::IKVStore::memory_handle_t _direct_memory_handle = Component::IKVStore::HANDLE_NONE;

public:
    ExperimentGetDirect(const ProgramOptions &options) : Experiment("get_direct", options) 
      , _i(0)
      , _start_time()
      , _latencies()
      , _exp_start_time()
      , _latency_stats()
    {    
    }
    ExperimentGetDirect(const ExperimentGetDirect &) = delete;
    ExperimentGetDirect& operator=(const ExperimentGetDirect &) = delete;

    void initialize_custom(unsigned core)
    {
      if (is_verbose())
      {
        PINF("[%u] exp_get_direct: initialize custom started", core);
      }

      _latency_stats = BinStatistics(bin_count(), bin_threshold_min(), bin_threshold_max());

      try
      {
        if ( component_is("dawn") )
        {
           size_t data_size = sizeof(KV_pair) * g_data->num_elements();
           Data * data = static_cast<Data*>(aligned_alloc(pool_size(), data_size));
           madvise(data, data_size, MADV_HUGEPAGE);
           _direct_memory_handle = store()->register_direct_memory(data, data_size);
        }
      }
      catch(...)
      {
        PERR("%s", "failed during direct_memory_handle setup");
        throw std::exception();
      }

      PLOG("%s", "pool seeded with values");
    }

    bool do_work(unsigned core) override 
    {
        // handle first time setup
        if(_first_iter) 
        {
          // seed the pool with elements from _data
          _populate_pool_to_capacity(core, _direct_memory_handle);

          wait_for_delayed_start(core);

          PLOG("[%u] Starting Get Direct experiment...", core);
          _first_iter = false;
          _exp_start_time = std::chrono::high_resolution_clock::now();
        }     

        // end experiment if we've reached the total number of components
        if (_i + 1 == pool_num_objects())
        {
          PINF("[%u] get_direct: reached total number of components. Exiting.", core);
          timer.stop();
          return false; 
        }

        // check time it takes to complete a single get_direct operation

        Component::io_buffer_t handle{};
        Core::Physical_memory mem_alloc;
        size_t pval_len = 64;
        void* pval = operator new(pval_len);
        Component::IKVStore::memory_handle_t memory_handle = Component::IKVStore::HANDLE_NONE; 

        if ( component_is("nvmestore") )
        {
            // TODO: make the input parameters 1 and 2 variable based on experiment inputs
            handle = mem_alloc.allocate_io_buffer(MB(8), 4096, Component::NUMA_NODE_ANY);
            pval = mem_alloc.virt_addr(handle);
            
            if (!handle)
            {
                perror("ExpGetDirect.do_work: allocate_io_buffer failed");
            }

            pval = mem_alloc.virt_addr(handle);
        }
        else if ( component_is("dawn") )
        {
            memory_handle = _direct_memory_handle;
        }

        {
          StopwatchInterval si(timer);
          auto rc = store()->get_direct(pool(), g_data->key(_i), pval, pval_len, memory_handle);
          if (rc != S_OK)
          {
              std::cout << "rc != S_OK: rc = " << rc << std::endl;
              throw std::exception();
          }
        }
       
        _update_data_process_amount(core, _i);

        double time = timer.get_lap_time_in_seconds();

        std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
        double time_since_start = double(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - _exp_start_time).count()) / 1000.0;
       
        if ( component_is("nvmestore") )
        { 
             mem_alloc.free_io_buffer(handle);
        }
        else if (pval != nullptr)
        {
          operator delete(pval);
        }

        // store the information for later use
        _latencies.push_back(time);
        _start_time.push_back(time_since_start);
        _latency_stats.update(time);
        

        ++_i;  // increment after running so all elements get used

       if (_i == std::size_t(pool_element_end()) + 1)
       {
            _erase_pool_entries_in_range(pool_element_start(), pool_element_end());
           _populate_pool_to_capacity(core, _direct_memory_handle);

           if ( is_verbose() )
           {
              std::stringstream debug_message;
              debug_message << "pool repopulated: " << _i;
              _debug_print(core, debug_message.str());
           }
       }
       return true;
    }

    void cleanup_custom(unsigned core)  
    {
        if ( component_is("dawn") )
        {
            store()->unregister_direct_memory(_direct_memory_handle);
        }

        double run_time = timer.get_time_in_seconds();
        double iops = double(_i) / run_time;
        PINF("[%u] get_direct: IOPS: %2g in %2g seconds", core, iops, run_time);
        _update_aggregate_iops(iops);

        double throughput = _calculate_current_throughput();
        PINF("[%u] get_direct: THROUGHPUT: %.2f MB/s (%lu bytes over %.3f seconds)", core, throughput, total_data_processed(), run_time);

        if ( is_verbose() )
        {
          std::ostringstream stats_info;
          stats_info << "creating time_stats with " << bin_count() << " bins: [" << _start_time.front() << " - " << _start_time.at(_i-1) << "]" << std::endl;
          _debug_print(core, stats_info.str());
        }

       // compute _start_time_stats pre-lock
       BinStatistics start_time_stats = _compute_bin_statistics_from_vectors(_latencies, _start_time, bin_count(), _start_time.front(), _start_time.at(_i-1), _i);

       if ( is_json_reporting() )
       {
         std::lock_guard<std::mutex> g(g_write_lock);

         // get existing results, read to document variable
         rapidjson::Document document = _get_report_document();

         // collect latency stats
         rapidjson::Value latency_object = _add_statistics_to_report("latency", _latency_stats, document);
         rapidjson::Value timing_object = _add_statistics_to_report("start_time", start_time_stats, document);
         rapidjson::Value iops_object;
         rapidjson::Value throughput_object;

         iops_object.SetDouble(iops);
         throughput_object.SetDouble(throughput);

         // save everything
         rapidjson::Value experiment_object(rapidjson::kObjectType);

         experiment_object.AddMember("IOPS", iops_object, document.GetAllocator());
         experiment_object.AddMember("throughput (MB/s)", throughput_object, document.GetAllocator());
         experiment_object.AddMember("latency", latency_object, document.GetAllocator());
         experiment_object.AddMember("start_time", timing_object, document.GetAllocator()); 
        _print_highest_count_bin(_latency_stats, core);

         _report_document_save(document, core, experiment_object);
       }
    }
};


#endif //  __EXP_GET_DIRECT_LATENCY_H__
