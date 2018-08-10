#ifndef __EXPERIMENT_H__
#define __EXPERIMENT_H__

#include <boost/filesystem.hpp>
#include <cstdio>
#include <fstream>
#include <ctime>

#include "data.h"
#include "kvstore_perf.h"

extern Data * _data;
extern int g_argc;
extern char** g_argv;

class Experiment : public Core::Tasklet
{ 
public:
    std::string _pool_path = "./data";
    unsigned int _pool_size = MB(100);
    int _pool_flags = Component::IKVStore::FLAGS_SET_SIZE;
    unsigned int _pool_num_components = 100;
    unsigned int _cores = 1;
    unsigned int _execution_time;
    std::string _component = "filestore";
    std::string _results_path = "./results";
    std::string _report_filename;
    std::string _test_name;

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
        sprintf(poolname, "Exp.pool.%u", core);
        PLOG("Creating pool for worker %u ...", core);
        _pool = _store->create_pool(_pool_path, poolname, _pool_size, _pool_flags, _pool_num_components);
      
        PLOG("Created pool for worker %u...OK!", core);
        
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

        _store->delete_pool(_pool);
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
          ("size", po::value<unsigned int>(), "Size of pool")
          ("flags", po::value<int>(), "Flags for pool creation")
          ("elements", po::value<unsigned int>(), "Number of data elements")
          ("key_length", po::value<unsigned int>(), "Key length of data")
          ("value_length", po::value<unsigned int>(), "Value length of data")
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
                _pool_size = vm["size"].as<unsigned int>();
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
        } 
        catch (const po::error &ex)
        {
          std::cerr << ex.what() << '\n';
        }
    }
    
    rapidjson::Document _get_report_document()
    {
       assert(!_report_filename.empty());  // make sure report_filename is set

       FILE *pFile = fopen(_report_filename.c_str(), "r+");
       assert(pFile);

       rapidjson::FileStream is(pFile);
       rapidjson::Document document;
       document.ParseStream<0>(is);

       return document;
    }

    void _report_document_save(rapidjson::Document& document, unsigned core, rapidjson::Value& new_info)
    {
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

        temp_value.SetInt(options.size);
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

    size_t                                _i = 0;
    Component::IKVStore *                 _store;
    Component::IKVStore::pool_t           _pool;
    bool                                  _first_iter = true;
    bool                                  _ready = false;
    std::chrono::system_clock::time_point _start, _end;
};


#endif //  __EXPERIMENT_H__
