#ifndef __EXPERIMENT_H__
#define __EXPERIMENT_H__

#include <boost/filesystem.hpp>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>
#include <vector>

#include "common/cycles.h"
#include "data.h"
#include "kvstore_perf.h"
#include "statistics.h"
#include "stopwatch.h"

extern Data * _data;
extern int g_argc;
extern char** g_argv;
extern pthread_mutex_t g_write_lock;
extern boost::program_options::options_description desc;

class Experiment : public Core::Tasklet
{ 
public:
  std::string _pool_path = "./data";
  std::string _pool_name = "Exp.pool.";
  unsigned long long int _pool_size = MB(100);
  int _pool_flags = Component::IKVStore::FLAGS_SET_SIZE;
  int _pool_num_components = 100000;
  std::string _cores = "0";
  int _execution_time;
  std::string _component = "filestore";
  std::string _results_path = "./results";
  std::string _report_filename;
  std::string _test_name;
  Component::IKVStore::memory_handle_t _memory_handle = Component::IKVStore::HANDLE_NONE;

  // common experiment parameters
  size_t                                _i = 0;
  Component::IKVStore *                 _store;
  Component::IKVStore::pool_t           _pool;
  bool                                  _first_iter = true;
  bool                                  _ready = false;
  Stopwatch timer;
  bool _verbose = false;
  bool _summary = false;

  // member variables for tracking pool sizes
  long _element_size = -1;
  long _elements_in_use = 0;
  long _pool_element_start = 0;
  long _pool_element_end = -1;
  long _elements_stored = 0;

  // bin statistics
  int _bin_count = 100;
  double _bin_threshold_min = 0.000000001;
  double _bin_threshold_max = 0.001;
  double _bin_increment;

  float _cycles_per_second = Core::get_rdtsc_frequency_mhz() * 1000000;

  Experiment(struct ProgramOptions options): _store(options.store)
  {
    assert(options.store);

    _report_filename = options.report_file_name;
  }
    
  void initialize(unsigned core) override 
  {
    handle_program_options();

    try
    {
      if (component_uses_direct_memory())
      {
        pthread_mutex_lock(&g_write_lock);

        size_t data_size = sizeof(Data) + ((sizeof(KV_pair) + _data->_key_len + _data->_val_len ) * _pool_num_components);
        data_size += get_difference_to_next_power_of_two(data_size);

        if (_verbose)
        {
          PINF("allocating %zu of aligned, direct memory. Aligned to %d", data_size,  MB(2));
        }

        _data->_data = (KV_pair*)aligned_alloc(MB(2), data_size);
        madvise(_data->_data, data_size, MADV_HUGEPAGE);

        _memory_handle = _store->register_direct_memory(_data->_data, data_size);
        _data->initialize_data(false);

        pthread_mutex_unlock(&g_write_lock);
      }
    }
    catch(...)
    {
      PERR("failed during direct memory setup");
      throw std::exception();
    }

    // make sure path is available for use
    boost::filesystem::path dir(_pool_path);
    if (boost::filesystem::create_directory(dir))
      {
        std::cout << "Created directory for testing: " << _pool_path << std::endl;
      }

    // initialize experiment
    char poolname[256];
    sprintf(poolname, "%s%u", _pool_name.c_str(), core);

    try
      {
        if (boost::filesystem::exists(_pool_path + "/" + poolname))
          {
            // pool already exists. Delete it.
            if (_verbose)
            {
              std::cout << "pool already exists at " << _pool_path + "/" + poolname << ". Attempting to delete it...";
            }

            _store->delete_pool(_store->open_pool(_pool_path, poolname));

            if (_verbose)
            {
              std::cout << " pool deleted!" << std::endl;
            }
          }
      }
    catch(...)
      {
        std::cerr << "open existing pool failed" << std::endl;
      }

    PLOG("Creating pool for worker %u ...", core);
    try
    {
      _pool = _store->create_pool(_pool_path, poolname, _pool_size, _pool_flags, _pool_num_components);
    }
    catch(...)
    {
      PERR("create_pool failed! Aborting experiment.");
      throw std::exception();
    }
      
    PLOG("Created pool for worker %u...OK!", core);

    try
    {
      initialize_custom(core);
    }
    catch(...)
    {
      PERR("initialize_custom failed! Aborting experiment.");
      throw std::exception();
    }

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
  bool do_work(unsigned core) override
  {
    assert(false && "Experiment.do_work needs to use override!");
    return false;
  }
   
  virtual void cleanup_custom(unsigned core)
  {
    // does nothing by itself; put per-experiment cleanup functions in its place
    printf("no cleanup_custom function used\n");
  }

  void cleanup(unsigned core) override 
  {
    try
    {
      cleanup_custom(core);
    }
    catch(...)
    {
      PERR("cleanup_custom failed!");
      throw std::exception();
    }

    try
    {
      if (component_uses_direct_memory())
      {
        _store->unregister_direct_memory(_memory_handle);
      }
    }
    catch(...)
    {
      PERR("unregister_direct_memory failed!");
      throw std::exception();
    }

    try
    {
      if (_verbose)
      {
        std::cout << "cleanup: attempting to delete pool" << std::endl;
      }

      _store->delete_pool(_pool);
    }
    catch(...)
    {
      PERR("delete_pool failed! Ending experiment.");
      throw std::exception();
    }
  }

  bool component_uses_direct_memory()
  {
    return _component.compare("dawn") == 0;
  }

  void handle_program_options()
  {
    namespace po = boost::program_options;
    ProgramOptions Options; 

    try 
      {
        po::variables_map vm; 
        po::store(po::parse_command_line(g_argc, g_argv, desc),  vm);

        if (vm.count("component") > 0) {
          _component = vm["component"].as<std::string>();
        }
        
        if(vm.count("path") > 0) {
          _pool_path = vm["path"].as<std::string>();
        }

        if(vm.count("size") > 0) {
          _pool_size = vm["size"].as<long long int>();
        }

        if (vm.count("elements") > 0) {
          _pool_num_components = vm["elements"].as<int>();
        }

        if (vm.count("flags") > 0) {
          _pool_flags = vm["flags"].as<int>();
        }

        if (vm.count("cores") > 0)  {
          _cores = vm["cores"].as<std::string>();
        }

        if (vm.count("bins") > 0) {
          _bin_count = vm["bins"].as<int>();
        }

        if (vm.count("latency_range_min") > 0) {
          _bin_threshold_min = vm["latency_range_min"].as<double>();
        }

        if (vm.count("latency_range_max") > 0) {
          _bin_threshold_max = vm["latency_range_max"].as<double>();
        }

        _verbose = vm.count("verbose");
        _summary = vm.count("summary");
      } 
    catch (const po::error &ex)
      {
        std::cerr << ex.what() << '\n';
      }
  }

  static cpu_mask_t get_cpu_mask_from_string(std::string core_string)
  {
    cpu_mask_t mask;
    unsigned cores_total = 0;
    unsigned core_first, core_last;
    int hardware_total_cores = std::thread::hardware_concurrency();

    int start = 0;
    int length = 0;
    int current = 0;
    std::string substring;
    std::string range_start = "";
    bool using_range = false;

    while(true)
    {
      if (core_string[current] == '-')
      {
        range_start = core_string.substr(start, length);

        start = current + 1;
        length = 0;

        using_range = !using_range;  // toggle
      }
      else if (core_string[current] == ',' || core_string[current] == '\0')
      {
        substring = core_string.substr(start, length);

        if (substring == "")
        {
          PERR("invalid core string. Tried to use '%s'. Exiting.", core_string.c_str());
          throw std::exception();
        }

        // if we were using a range, substring becomes the end range value
        core_last = (unsigned)std::stoi(substring);

        if (using_range)
        {
          if (range_start == "")
          {
            PERR("no start core specified for range ending with %u", core_last);
            throw std::exception();
          }

          core_first = (unsigned)std::stoi(range_start);
          using_range = false;
          range_start = "";
        }
        else // single core add
        {
          core_first = core_last;
        }

        _cpu_mask_add_core_wrapper(mask, core_first, core_last, hardware_total_cores);

        start = current + 1;
        length = 0;

        if (core_string[current] == '\0')
        {
          break;
        }
      }
      else
      {
        length++;
      }

      current++;
    }

    return mask;
  }

  static void _cpu_mask_add_core_wrapper(cpu_mask_t &mask, unsigned core_first, unsigned core_last, unsigned max_cores)
  {
    if (core_first > core_last)
    {
      PERR("invalid core range specified: start (%u) > end (%u).", core_first, core_last);
      throw std::exception();
    }
    else if (core_first > max_cores - 1 || core_last > max_cores - 1)  // max_cores is zero indexed
    {
      PERR("specified core range (%u-%u) exceeds physical core count. Valid range is 0-%d.", core_first, core_last, max_cores - 1);
      throw std::exception();
    }
    else if (core_first < 0 || core_last < 0)
    {
      PERR("core values must be positive! Trying to use %u-%u.", core_first, core_last);
    }

    try
    {
      for (unsigned core = core_first; core <= core_last; core++)
      {
        mask.add_core(core);
      }
    }
    catch(...)
    {
      PERR("failed while adding core to mask.");
      throw std::exception();
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
    rapidjson::Document document;

    if (_report_filename.empty())
    {
      PERR("filename for report is empty!");
      throw std::exception();
    }

    try
    {
      FILE *pFile = fopen(_report_filename.c_str(), "r");
      if (!pFile)
        {
          std::cerr << "attempted to open filename '" << _report_filename << "'" << std::endl;
          perror("_get_report_document failed fopen call");
          throw std::exception();
        }

      char readBuffer[GetFileSize(_report_filename)];

      rapidjson::FileReadStream is(pFile, readBuffer, sizeof(readBuffer));
      document.ParseStream<0>(is);

      if (document.HasParseError())
      {
        PERR("parsing error in document, code = %d", (int)document.GetParseError());
        throw std::exception();
      }

      fclose(pFile);
    }
    catch(...)
    {
      PERR("failed while reading in existing json document");
      throw std::exception();
    }

    _debug_print(0, "returning report document");
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

    if (_test_name.empty())
    {
      PERR("_test_name is empty!");
      throw std::exception();
    }

    rapidjson::Value temp_value;
    rapidjson::Value temp_object(rapidjson::kObjectType);
    rapidjson::StringBuffer strbuf;

    std::string core_string = std::to_string(core);
    temp_value.SetString(rapidjson::StringRef(core_string.c_str()));

    try
    {
      if (document.IsObject() && !document.HasMember(_test_name.c_str()))
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
      rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(strbuf);
      document.Accept(writer);
    }
    catch(...)
    {
      PERR("failed during write to json document");
    }

    _debug_print(core, "_report_document_save: writing to ofstream");
    try
    {
      std::ofstream outf(_report_filename.c_str());
      outf << strbuf.GetString() << std::endl;
    }
    catch(...)
    {
      PERR("failed while writing to ofstream");
      throw std::exception();
    }

    _debug_print(core, "_report_document_save finished");
  }

  void _print_highest_count_bin(BinStatistics& stats, unsigned core)
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

    if (count_highest > -1 && _summary)
      {
        RunningStatistics bin = stats.getBin(count_highest_index);

        // print information about that bin
        std::cout << "SUMMARY: core " << core << std::endl;
        std::cout << "\tmean:\t" << bin.getMean() << std::endl;
        std::cout << "\tmin:\t" << bin.getMin() << std::endl;
        std::cout << "\tmax:\t" << bin.getMax() << std::endl;
        std::cout << "\tstd:\t" << bin.getMax() << std::endl;
        std::cout << "\tcount:\t" << bin.getCount() << std::endl;
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
    PLOG("creating JSON report");
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

    temp_value.SetString(rapidjson::StringRef(options.cores.c_str()));
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
    if (_verbose)
    {
      std::cout << "_populate_pool_to_capacity start: _pool_num_components = " << _pool_num_components << ", _elements_stored = " << _elements_stored << ", _pool_element_end = " << _pool_element_end << std::endl;
    }

    long elements_remaining = _pool_num_components - _elements_stored;
    bool can_add_more_elements;
    int rc;
    long current = _pool_element_end + 1;  // first run: should be 0 (start index)
    long maximum_elements = -1;
    _pool_element_start = current;
      
    if (_verbose)
    { 
      std::stringstream debug_start;
      debug_start << "current = " << current << ", end = " << _pool_element_end;
      _debug_print(core, debug_start.str());
    }

    do
    {
      try
      {
        if (memory_handle != Component::IKVStore::HANDLE_NONE)
        {
          rc = _store->put_direct(_pool, _data->key(current), _data->value(current), _data->_val_len, memory_handle);
        }
        else
        {
          rc = _store->put(_pool, _data->key(current), _data->value(current), _data->value_len());
        }

        _elements_stored++;
      }
      catch(...)
      {
        std::cerr << "current = " << current << std::endl;
        PERR("_populate_pool_to_capacity failed at put call");
        throw std::exception();
      }

      if (rc != S_OK)
      {
        std::cerr << "current = " << current << std::endl;
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
        maximum_elements = (long)(_pool_size / _element_size);

        if (_verbose)
        {
          std::stringstream debug_element_max;
          debug_element_max << "maximum element count: " << maximum_elements;
          _debug_print(core, debug_element_max.str());
        }
      }

      current++;

      bool can_add_more_in_batch = (current - _pool_element_start) != maximum_elements;
      bool can_add_more_overall = current != _pool_num_components;

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

    _pool_element_end = current - 1;

    if (_verbose)
    {
      std::cout << "_pool_element_end = " << _pool_element_end << std::endl;
      std::stringstream range_info;
      range_info << "current = " << current << ", end = " << _pool_element_end;
      _debug_print(core, range_info.str(), true);

      range_info = std::stringstream();
      range_info << "elements added to pool: " << current - _pool_element_start << ". Last = " << current;
      _debug_print(core, range_info.str(), true);
    }
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

      try
      {
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
      }
      catch(...)
      {
        PERR("failed during erase step");
        throw std::exception();
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
    if (_verbose)
    {
      std::cout << "erasing pool entries in range " << start << " to " << finish << std::endl;
    }

    int rc;

    try
    {
      for (int i = start; i < finish; i++)
      {
        rc = _store->erase(_pool, _data->key(i));

        if (rc != S_OK)
          {
            throw std::exception();
          }
      }
    }
    catch(...)
    {
      PERR("erase step failed");
      throw std::exception();
    }
  }

  static size_t get_difference_to_next_power_of_two(size_t value)
  {
    if (value < 0)
    {
      PERR("get_difference_to_next_power_of_two only supports positive numbers");
      throw std::exception();
    }

    if (value == 0 || value == 1)
    {
      return 0;
    }

    int i = 1;
    std::vector<size_t> powers_of_two = {1, 2};

    while (value > powers_of_two[i])
    {
      i++;
      powers_of_two.push_back(std::pow(2, i));
    }

    return powers_of_two[i] - value;
  }
};


#endif //  __EXPERIMENT_H__
