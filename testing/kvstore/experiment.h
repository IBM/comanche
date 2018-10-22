#ifndef __EXPERIMENT_H__
#define __EXPERIMENT_H__

#include <boost/filesystem.hpp>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>

#include "data.h"
#include "kvstore_perf.h"
#include "statistics.h"
#include "stopwatch.h"

extern Data * _data;
extern int g_argc;
extern char** g_argv;

class Experiment : public Core::Tasklet
{ 
public:
    std::string _pool_path = "./data";
    std::string _pool_name = "Exp.pool.";
    unsigned long long int _pool_size = MB(100);
    int _pool_flags = Component::IKVStore::FLAGS_SET_SIZE;
    unsigned int _pool_num_components = 100000;
    unsigned int _cores = 1;
    unsigned int _execution_time;
    std::string _component = "filestore";
    std::string _results_path = "./results";
    std::string _report_filename;
    std::string _test_name;
    Component::IKVStore::memory_handle_t _memory_handle = Component::IKVStore::HANDLE_NONE;

    Experiment(struct ProgramOptions options): _store(options.store)
    {
        assert(options.store);

        _report_filename = options.report_file_name;
    }
    
    void initialize(unsigned core) override 
    {
        handle_program_options();

        // make sure path is available for use
        boost::filesystem::path dir(_pool_path);
        if (boost::filesystem::create_directory(dir))
        {
            std::cout << "Created directory for testing: " << _pool_path << std::endl;
        }

        // initialize experiment
        char poolname[256];
        sprintf(poolname, "%s%u", _pool_name.c_str(), core);
        PLOG("Creating pool for worker %u ...", core);
        _pool = _store->create_pool(_pool_path, poolname, _pool_size, _pool_flags, _pool_num_components);
      
        PLOG("Created pool for worker %u...OK!", core);

        if (component_uses_direct_memory())
        {
           size_t data_size = sizeof(KV_pair) * _data->_num_elements;
           data_size += data_size % 64;  // align
           _data->_data = (KV_pair*)aligned_alloc(MiB(2), data_size);
           madvise(_data->_data, data_size, MADV_HUGEPAGE);

           _memory_handle = _store->register_direct_memory(_data->_data, data_size);
           _data->initialize_data(false);
        }

        initialize_custom(core);

        ProfilerRegisterThread();
        _ready = true;
    };

    virtual void initialize_custom(unsigned core)
    {
        // does nothing by itself; put per-experiment initialization functions here
        printf("no initialize_custom function used\n");
    }

    bool ready() override 
    {
        return _ready;
    }

    // do_work should be overwritten by child class
    void do_work(unsigned core) override
    {
        assert(false && "Experiment.do_work needs to use override!");
    }
   
    virtual void cleanup_custom(unsigned core)
    {
        // does nothing by itself; put per-experiment cleanup functions in its place
        printf("no cleanup_custom function used\n");
    }

    void cleanup(unsigned core) override 
    {
        cleanup_custom(core);

        if (component_uses_direct_memory())
        {
            _store->unregister_direct_memory(_memory_handle);
        }

        _store->delete_pool(_pool);
    }

    bool component_uses_direct_memory()
    {
        return _component.compare("dawn_client") == 0;
    }

    bool should_free_memory_after_get()
    {
        return _component.compare("mapstore") != 0;  // should be true when not mapstore
    }

    void handle_program_options()
    {
        namespace po = boost::program_options;
        ProgramOptions Options; 

        po::options_description desc("Options"); 
        desc.add_options()
          ("help", "Show help")
          ("test", po::value<std::string>(), "Test name <all|Put|Get>")
          ("component", po::value<std::string>(), "Implementation selection <pmstore|nvmestore|filestore>")
          ("cores", po::value<int>(), "Number of threads/cores")
          ("time", po::value<int>(), "Duration to run in seconds")
          ("path", po::value<std::string>(), "Path of directory for pool")
          ("size", po::value<unsigned long long int>(), "Size of pool")
          ("flags", po::value<int>(), "Flags for pool creation")
          ("elements", po::value<unsigned int>(), "Number of data elements")
          ("key_length", po::value<unsigned int>(), "Key length of data")
          ("value_length", po::value<unsigned int>(), "Value length of data")
          ("bins", po::value<unsigned int>(), "Number of bins for statistics")
          ("latency_range_min", po::value<double>(), "Lowest latency bin threshold")
          ("latency_range_max", po::value<double>(), "Highest latency bin threshold")
          ("debug_level", po::value<int>(), "Debug level")
          ("owner", po::value<std::string>(), "Owner name for component registration")
          ("server_address", po::value<std::string>(), "server address, with port")
          ("device_name", po::value<std::string>(), "device name")
                  ;
      
        try 
        {
            po::variables_map vm; 
            po::store(po::parse_command_line(g_argc, g_argv, desc),  vm);

            if (vm.count("component") > 0)
            {
                _component = vm["component"].as<std::string>();
            }

            if(vm.count("path") > 0)
            {
                _pool_path = vm["path"].as<std::string>();
            }

            if(vm.count("size") > 0)
            {
                _pool_size = vm["size"].as<unsigned long long int>();
            }

            if (vm.count("elements") > 0)
            {
                _pool_num_components = vm["elements"].as<unsigned int>();
            }

            if (vm.count("flags") > 0)
            {
                _pool_flags = vm["flags"].as<int>();
            }

            if (vm.count("cores") > 0)
            {
                _cores = vm["cores"].as<int>();
            }

            if (vm.count("bins") > 0)
            {
                _bin_count = vm["bins"].as<unsigned int>();
            }

            if (vm.count("latency_range_min") > 0)
            {
                _bin_threshold_min = vm["latency_range_min"].as<double>();
            }

            if (vm.count("latency_range_max") > 0)
            {
                _bin_threshold_max = vm["latency_range_max"].as<double>();
            }
        } 
        catch (const po::error &ex)
        {
          std::cerr << ex.what() << '\n';
        }
    }
   
    void _debug_print(unsigned core, std::string text, bool limit_to_core0=false)
    {
        if (_verbose)
        {
            if (limit_to_core0 && core==0 || !limit_to_core0)
            {
                std::cout << "[" << core << "]: " << text << std::endl;
            }
        }
    }

    rapidjson::Document _get_report_document()
    {
       assert(!_report_filename.empty());  // make sure report_filename is set

       FILE *pFile = fopen(_report_filename.c_str(), "r+");
       if (!pFile)
       {
           perror("_get_report_document failed fopen call");
           throw std::exception();
       }

       char readBuffer[GetFileSize(_report_filename)];

       rapidjson::FileReadStream is(pFile, readBuffer, sizeof(readBuffer));
       rapidjson::Document document;
       document.ParseStream<0>(is);

       return document;
    }

    long GetFileSize(std::string filename)
    {
        struct stat stat_buf;

        int rc = stat(filename.c_str(), &stat_buf);

        return rc == 0 ? stat_buf.st_size : -1;
    }

    long GetBlockSize(std::string path)
    {
        struct stat stat_buf;

        int rc = stat(path.c_str(), &stat_buf);

        return rc == 0 ? stat_buf.st_blksize : -1;
    }

    void _report_document_save(rapidjson::Document& document, unsigned core, rapidjson::Value& new_info)
    {
       _debug_print(core, "_report_document_save started");

       assert(!_test_name.empty());  // make sure _test_name is set

       rapidjson::Value temp_value;
       rapidjson::Value temp_object(rapidjson::kObjectType);

       std::string core_string = std::to_string(core);
       temp_value.SetString(rapidjson::StringRef(core_string.c_str()));

       if (!document.HasMember(_test_name.c_str()))
       {
           temp_object.AddMember(temp_value, new_info, document.GetAllocator());
            document.AddMember(rapidjson::StringRef(_test_name.c_str()), temp_object, document.GetAllocator()); 
       }
       else 
       {
            rapidjson::Value &items = document[_test_name.c_str()];
            &items.AddMember(temp_value, new_info, document.GetAllocator());
       }

        // write back to file
       rapidjson::StringBuffer strbuf;
       rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
       document.Accept(writer);

       std::ofstream outf(_report_filename.c_str());
       outf << strbuf.GetString() << std::endl;

       _debug_print(core, "_report_document_save finished");
    }

    void _print_highest_count_bin(BinStatistics& stats)
    {
        int count_highest = -1;  // arbitrary placeholder value
        int count_highest_index = -1;  // arbitrary placeholder value

        // find bin with highest count
        for (int i = 0; i < stats.getBinCount(); i++)
        {
            if (stats.getBin(i).getCount() > count_highest)
            {
                count_highest = stats.getBin(i).getCount();
                count_highest_index = i;
            }
        }

        if (count_highest > -1)
        {
            RunningStatistics bin = stats.getBin(count_highest_index);

            // print information about that bin
            std::cout << "SUMMARY: " << std::endl;
            std::cout << "\tmean: " << bin.getMean() << std::endl;
            std::cout << "\tmin: " << bin.getMin() << std::endl;
            std::cout << "\tmax: " << bin.getMax() << std::endl;
            std::cout << "\tstd: " << bin.getMax() << std::endl;
            std::cout << "\tcount: " << bin.getCount() << std::endl;
        }
    }

    rapidjson::Value _add_statistics_to_report(std::string name, BinStatistics& stats, rapidjson::Document& document)
    {
       rapidjson::Value bin_info(rapidjson::kObjectType);
       rapidjson::Value temp_array(rapidjson::kArrayType);
       rapidjson::Value temp_value;

       // latency bin info
       temp_value.SetInt(stats.getBinCount());
       bin_info.AddMember("bin_count", temp_value, document.GetAllocator());

       temp_value.SetDouble(stats.getMinThreshold());
       bin_info.AddMember("threshold_min", temp_value, document.GetAllocator());

       temp_value.SetDouble(stats.getMaxThreshold());
       bin_info.AddMember("threshold_max", temp_value, document.GetAllocator());

       temp_value.SetDouble(stats.getIncrement());
       bin_info.AddMember("increment", temp_value, document.GetAllocator());

       for (int i = 0; i < stats.getBinCount(); i++)  
       {
            // PushBack requires unique object
            rapidjson::Value temp_object(rapidjson::kObjectType); 

            temp_value.SetDouble(stats.getBin(i).getCount());
            temp_object.AddMember("count", temp_value, document.GetAllocator());

            temp_value.SetDouble(stats.getBin(i).getMin());
            temp_object.AddMember("min", temp_value, document.GetAllocator());

            temp_value.SetDouble(stats.getBin(i).getMax());
            temp_object.AddMember("max", temp_value, document.GetAllocator());

            temp_value.SetDouble(stats.getBin(i).getMean());
            temp_object.AddMember("mean", temp_value, document.GetAllocator());

            temp_value.SetDouble(stats.getBin(i).getStd());
            temp_object.AddMember("std", temp_value, document.GetAllocator());

            temp_array.PushBack(temp_object, document.GetAllocator());
       }

       // add new info to report
       rapidjson::Value bin_object(rapidjson::kObjectType);
      
       bin_object.AddMember("info", bin_info, document.GetAllocator());
       bin_object.AddMember("bins", temp_array, document.GetAllocator());

       return bin_object;
    }

    static std::string get_time_string()
    {
        time_t rawtime;
        struct tm *timeinfo;
        time(&rawtime); 
        timeinfo = localtime(&rawtime);
        char buffer[80];

        // want YYYY_MM_DD_HH_MM format
        strftime(buffer, sizeof(buffer), "%Y_%m_%d_%H_%M", timeinfo);
        std::string timestring(buffer);

        return timestring;
    }

    BinStatistics _compute_bin_statistics_from_vectors(std::vector<double> data, std::vector<double> data_bins, int bin_count, double bin_min, double bin_max, int elements)
    {
        if (data.size() != data_bins.size())
        {
            perror("data and data_bins sizes aren't the same!");
        }

        BinStatistics stats(bin_count, bin_min, bin_max);

        for (int i = 0; i < elements; i++)
        {
            stats.update_value_for_bin(data[i], data_bins[i]);
        }

        return stats;
    }

    BinStatistics _compute_bin_statistics_from_vector(std::vector<double> data, int bin_count, double bin_min, double bin_max)
    {
        BinStatistics stats(bin_count, bin_min, bin_max);

        for (int i = 0; i < data.size(); i++)
        {
            stats.update(data[i]);
        }

        return stats;
    }

    /* create_report: output a report in JSON format with experiment data
     * Report format: 
     *      experiment object - contains experiment parameters 
     *      data object - actual results
     */ 
    static std::string create_report(ProgramOptions options)
    {
        std::string timestring = get_time_string();

        // create json document/object
        rapidjson::Document document;
        document.SetObject();

        rapidjson::Document::AllocatorType &allocator = document.GetAllocator();

        rapidjson::Value temp_object(rapidjson::kObjectType);
        rapidjson::Value temp_value;

        // experiment parameters
        temp_value.SetString(rapidjson::StringRef(options.component.c_str()));
        temp_object.AddMember("component", temp_value, allocator);

        temp_value.SetInt(options.cores);
        temp_object.AddMember("cores", temp_value, allocator);

        temp_value.SetInt(options.key_length);
        temp_object.AddMember("key_length", temp_value, allocator);

        temp_value.SetInt(options.value_length);
        temp_object.AddMember("value_length", temp_value, allocator);

        temp_value.SetInt(options.elements);
        temp_object.AddMember("elements", temp_value, allocator);

        temp_value.SetDouble(options.size);
        temp_object.AddMember("pool_size", temp_value, allocator);

        temp_value.SetInt(options.flags);
        temp_object.AddMember("pool_flags", temp_value, allocator);

        temp_value.SetString(rapidjson::StringRef(timestring.c_str()));
        temp_object.AddMember("date", temp_value, allocator);

        document.AddMember("experiment", temp_object, allocator);


        // write to file
        std::string results_path = "./results";
        boost::filesystem::path dir(results_path);
        if (boost::filesystem::create_directory(dir))
        {
            std::cout << "Created directory for testing: " << results_path << std::endl;
        }

        std::string specific_results_path = results_path;
        specific_results_path.append("/" + options.component);

        boost::filesystem::path sub_dir(specific_results_path);
        if (boost::filesystem::create_directory(sub_dir))
        {
            std::cout << "Created directory for testing: " << specific_results_path << std::endl;
        }

        rapidjson::StringBuffer sb;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(sb);
        document.Accept(writer); 

        std::string output_file_name = specific_results_path + "/results_" + timestring + ".json"; 
        std::ofstream outf(output_file_name);

        if (!outf)
        {
            std::cerr << "couldn't open report file to write. Exiting.\n";
        }

        outf << sb.GetString() << std::endl;

        return output_file_name;
    }

    unsigned long GetElementSize(unsigned core, int index)
    {
        if (_element_size == -1)  // -1 is reserved value and impossible (must be positive size)
        {
            std::string path = _pool_path + "/" +  _pool_name + std::to_string(core) + "/" + _data->key(index);
            long block_size = GetBlockSize(_pool_path);
            long file_size = GetFileSize(path);

            // take the larger of the two
            if (file_size > block_size)
            {
                _element_size = file_size;
            }
            else
            {
                _element_size = block_size;
            }
        }

        return _element_size;
    }

    void _populate_pool_to_capacity(unsigned core, Component::IKVStore::memory_handle_t memory_handle = Component::IKVStore::HANDLE_NONE)
    {
        // how much space do we have?
        unsigned long elements_remaining = _pool_num_components - _elements_stored;
        bool can_add_more_elements;
        int rc;
        unsigned long current = _pool_element_end;  // first run: should be 0 (start index)
        unsigned long maximum_elements = -1;
        size_t offset = 0;
        _pool_element_start = current;
      
        if (_verbose)
        { 
            std::stringstream debug_start;
            debug_start << "current = " << current << ", end = " << _pool_element_end;
            _debug_print(core, debug_start.str());
        }

        do
        {
            if (memory_handle != Component::IKVStore::HANDLE_NONE)
            {
                rc = _store->put_direct(_pool, _data->key(current), _data->value(current), _data->_val_len, offset, memory_handle);
            }
            else
            {
                rc = _store->put(_pool, _data->key(current), _data->value(current), _data->value_len());
            }

            if (rc != S_OK)
            {
               perror("rc didn't return S_OK");
               throw std::exception(); 
            }

            // calculate maximum number of elements we can put in pool at one time
            if (_element_size == -1)
            {
                _element_size = GetElementSize(core, current);

                if (_verbose)
                {
                    std::stringstream debug_element_size;
                    debug_element_size << "element size is " << _element_size;
                    _debug_print(core, debug_element_size.str());
                }
            }

            if (maximum_elements == -1)
            {
                maximum_elements = (unsigned long)(_pool_size / _element_size);

                if (_verbose)
                {
                    std::stringstream debug_element_max;
                    debug_element_max << "maximum element count: " << maximum_elements;
                    _debug_print(core, debug_element_max.str());
                }
            }

            current++;

            bool can_add_more_in_batch = (current - _pool_element_start) <= maximum_elements;
            bool can_add_more_overall = current <= (_pool_num_components - 1);

           can_add_more_elements = can_add_more_in_batch && can_add_more_overall;

            if (!can_add_more_elements)
            {
                if (!can_add_more_in_batch)
                {
                    _debug_print(core, "reached capacity", true);
                }

                if (!can_add_more_overall)
                {
                    _debug_print(core, "reached last element", true);
                }
            }
        }
        while(can_add_more_elements);

        if (_verbose)
        {
            std::stringstream range_info;
            range_info << "current = " << current << ", end = " << _pool_element_end;
            _debug_print(core, range_info.str(), true);

            range_info = std::stringstream();
            range_info << "elements added to pool: " << current - _pool_element_end << ". Last = " << current;
            _debug_print(core, range_info.str(), true);
        }
        _pool_element_end = current;
    }

    // assumptions: _i is tracking current element in use
    void _enforce_maximum_pool_size(unsigned core)
    {
        unsigned long block_size = GetElementSize(core, _i);

        _elements_in_use++;

        // erase elements that exceed pool capacity and start again
        if ((_elements_in_use * _element_size) >= _pool_size)
        {
            bool timer_running_at_start = timer.is_running();  // if timer was running, pause it

            if (timer_running_at_start)
            {
                timer.stop();
            }

            if(_verbose)
            {
                std::stringstream debug_message;
                debug_message << "exceeded acceptable pool size. Erasing " << _elements_in_use << " elements...";

                _debug_print(core, debug_message.str(), true);
            }

            for (int i = _i - 1; i > (_i - _elements_in_use); i--)
            {
                int rc =_store->erase(_pool, _data->key(i));
                if (rc != S_OK && core == 0)
                {
                    // throw exception
                    std::string error_string = "erase returned !S_OK: ";
                    error_string.append(std::to_string(rc));
                    error_string.append(", i = " + std::to_string(i) + ", _i = " + std::to_string(_i));
                    perror(error_string.c_str());
                }                 
            }

            _elements_in_use = 0;   

            if (_verbose)
            {
                std::stringstream debug_end;
                debug_end << "done. _i = " << _i;

                _debug_print(core, debug_end.str(), true);
            }

            if (timer_running_at_start)
            {
                timer.start();
            }
        }
    }

    void _erase_pool_entries_in_range(int start, int finish)
    {
        int rc;

        for (int i = start; i < finish; i++)
        {
            rc = _store->erase(_pool, _data->key(i));

            if (rc != S_OK)
            {
                throw std::exception();
            }
        }
    }

    size_t                                _i = 0;
    Component::IKVStore *                 _store;
    Component::IKVStore::pool_t           _pool;
    bool                                  _first_iter = true;
    bool                                  _ready = false;
    std::chrono::system_clock::time_point _start, _end;
    Stopwatch timer;
    bool _verbose = true;

    // member variables for tracking pool sizes
    unsigned long _element_size = -1;
    unsigned long _elements_in_use = 0;
    unsigned long _pool_element_start = 0;
    unsigned long _pool_element_end = 0;
    unsigned long _elements_stored = 0;

    // bin statistics
    unsigned int _bin_count = 100;
    double _bin_threshold_min = (1. * std::pow(10, -9));
    double _bin_threshold_max = (1. * std::pow(10, -3));
    double _bin_increment;
};


#endif //  __EXPERIMENT_H__
