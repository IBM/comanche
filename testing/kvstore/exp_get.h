#ifndef __EXP_GET_H__
#define __EXP_GET_H__

#include "data.h"

extern Data * _data;


class Experiment_Get : public Core::Tasklet
{ 
public:

  Experiment_Get(Component::IKVStore * arg) : _store(arg) {
    assert(arg);
  }
  
  void initialize(unsigned core) override {
    char poolname[256];
    sprintf(poolname, "Get.pool.%u", core);
    PLOG("Creating pool for worker %u ...", core);
    _pool = _store->create_pool("/mnt/pmem0", poolname, GB(4),
                                Component::IKVStore::FLAGS_SET_SIZE, 100000);
    
    PLOG("(%u) Populating key/value pairs for Get test...", core);
    for(size_t i=0;i<_data->num_elements();i++) {
      int rc = _store->put(_pool, _data->key(i), _data->value(i), _data->value_len());
      if(rc != S_OK)
        PERR("store->put return code: %d", rc);
      //      assert(rc == S_OK);
    }
    PLOG("...OK");
    ProfilerRegisterThread();
    _ready = true;
  };

  bool ready() override {
    return _ready;
  }
  
  void do_work(unsigned core) override {
    if(_first_iter) {
      PLOG("Starting Get experiment...");
      _start = std::chrono::high_resolution_clock::now();
      _first_iter = false;
    }

    if(_i == _data->num_elements()) throw std::exception();

    void * pval;
    size_t pval_len;
    
    int rc = _store->get(_pool, _data->key(_i), pval, pval_len);
    assert(rc == S_OK);
    free(pval);
    _i++;
  }
  
  void cleanup(unsigned core) override {
    _end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() / 1000.0;
    PINF("*Get* (%u) IOPS: %lu", core, (uint64_t) (((double) _i) / secs));

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


#endif // __EXP_GET_H__
