#ifndef __EXPERIMENT_H__
#define __EXPERIMENT_H__

#include <boost/filesystem.hpp>

#include "data.h"

extern Data * _data;

class Experiment : public Core::Tasklet
{ 
public:
    std::string _pool_path = "./data";
    unsigned int _pool_size = MB(100);
    int _pool_flags = Component::IKVStore::FLAGS_SET_SIZE;
    unsigned int _pool_num_components = 100;
  
    Experiment(Component::IKVStore * arg) : _store(arg) 
    {
        assert(arg);
    }

    Experiment(Component::IKVStore *store, std::string path, 
            unsigned int size, int flags, unsigned int components):
        _store(store),
        _pool_path(path),
        _pool_size(size),
        _pool_flags(flags),
        _pool_num_components(components)
    {
        assert(store);
    }

    void initialize(unsigned core) override 
    {
        // make sure path is available for use
        boost::filesystem::path dir(_pool_path);
        if (boost::filesystem::create_directory(dir))
        {
            std::cout << "Created directory for testing: " << _pool_path << std::endl;
        }

        // initialize experiment
        printf("Experiment initialize\n");
        char poolname[256];
        sprintf(poolname, "Exp.pool.%u", core);
        PLOG("Creating pool for worker %u ...", core);
        _pool = _store->create_pool(_pool_path, poolname, _pool_size, _pool_flags, _pool_num_components);
      
        PLOG("Created pool for worker %u ...OK!", core);
        //    _pool = _store->open_pool("/dev/", "dax0.0");
        
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
  
    size_t                                _i = 0;
    Component::IKVStore *                 _store;
    Component::IKVStore::pool_t           _pool;
    bool                                  _first_iter = true;
    bool                                  _ready = false;
    std::chrono::system_clock::time_point _start, _end;
};


#endif //  __EXPERIMENT_H__
