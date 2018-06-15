#ifndef __EXP_PUT_H__
#define __EXP_PUT_H__

#include "data.h"

extern Data * _data;

class Experiment_Put : public Core::Tasklet
{ 
public:

  Experiment_Put(Component::IKVStore * arg) : _store(arg) {
    assert(arg);
  }
  
  void initialize(unsigned core) override {
    char poolname[256];
    sprintf(poolname, "Put.pool.%u", core);
    PLOG("Creating pool for worker %u ...", core);
    _pool = _store->create_pool("/mnt/pmem0", poolname, GB(1),
                                Component::IKVStore::FLAGS_SET_SIZE, 100000);
    
    PLOG("Created pool for worker %u ...OK!", core);
    //    _pool = _store->open_pool("/dev/", "dax0.0");
    ProfilerRegisterThread();
    _ready = true;
  };

  bool ready() override {
    return _ready;
  }

  void do_work(unsigned core) override {
    if(_first_iter) {
      _start = std::chrono::high_resolution_clock::now();
      _first_iter = false;
    }
    
    _i++;
    int rc = _store->put(_pool, _data->key(_i), _data->value(_i), _data->value_len());
    assert(rc == S_OK);
  }
  
  void cleanup(unsigned core) override {
    _end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / 1000.0;
    PINF("*Put* (%u) IOPS: %2g", core, ((double) _i) / secs);

    _store->delete_pool(_pool);
  }

private:
  size_t                                _i = 0;
  Component::IKVStore *                 _store;
  Component::IKVStore::pool_t           _pool;
  bool                                  _first_iter = true;
  bool                                  _ready = false;
  std::chrono::system_clock::time_point _start, _end;
};


#endif //  __EXP_PUT_H__
