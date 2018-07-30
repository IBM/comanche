#ifndef __EXPERIMENT_H__
#define __EXPERIMENT_H__

#include <boost/filesystem.hpp>

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
  
    Experiment(Component::IKVStore * arg) : _store(arg) 
    {
        assert(arg); 
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
          ;
      
        try 
        {
            po::variables_map vm; 
            po::store(po::parse_command_line(g_argc, g_argv, desc),  vm);

            if(vm.count("path"))
            {
                _pool_path = vm["path"].as<std::string>();
            }

            if(vm.count("size"))
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
        } 
        catch (const po::error &ex)
        {
          std::cerr << ex.what() << '\n';
        }
    }
  
    size_t                                _i = 0;
    Component::IKVStore *                 _store;
    Component::IKVStore::pool_t           _pool;
    bool                                  _first_iter = true;
    bool                                  _ready = false;
    std::chrono::system_clock::time_point _start, _end;
};


#endif //  __EXPERIMENT_H__
