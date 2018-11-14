#ifndef __DAWN_SHARD_H__
#define __DAWN_SHARD_H__

#ifdef __cplusplus

#include <thread>
#include <map>
#include <api/components.h>
#include <api/fabric_itf.h>
#include <api/kvstore_itf.h>
#include <api/cluster_itf.h>
#include <common/logging.h>
#include <common/exceptions.h>
#include <common/cpu.h>
#include "dawn_config.h"
#include "program_options.h"
#include "connection_handler.h"

#ifdef CONFIG_TRANSPORT_RDMA
#include "rdma_transport.h"
#else
#include "fabric_transport.h"
#endif

namespace Dawn {

class Connection_handler;

/* Adapter point */

using Shard_transport = Fabric_transport;

class Shard : public Shard_transport
{
private:
  using memory_region_t = Shard_transport::memory_region_t;
  using buffer_t = Shard_transport::buffer_t; 
  using pool_t = Component::IKVStore::pool_t;

  bool option_DEBUG;

public:
  Shard(Program_options& po,
        bool forced_exit) :
    Shard_transport(po.fabric_provider, po.device, po.port),
    _data_dir(po.data_dir),
    _core(po.core),
    _thread(&Shard::thread_entry, this, po),
    _forced_exit(forced_exit)
  {
    Dawn::Global::debug_level = po.debug_level;
    option_DEBUG = Dawn::Global::debug_level > 1;

    /* check data dir write access */
    if(access(po.data_dir.c_str(), W_OK) != 0)
      throw General_exception("data directory (%s) not writable",
                              po.data_dir.c_str());
  }

  ~Shard() {
    _thread_exit = true;
    /* TODO: unblock */
    _thread.join();
    _i_kvstore->release_ref();
    //    delete _i_fabric_factory;
  }

  bool exited() const { return _thread_exit; }
private:

  void thread_entry(const Program_options& po) {
    if(option_DEBUG)
      PLOG("shard:%u worker thread entered.", _core);

    if(set_cpu_affinity(1UL << _core) != 0)
      throw General_exception("unable to set cpu affinity (%lu)", _core);

    initialize_components(po);

    main_loop();

    if(option_DEBUG)
      PLOG("shard:%u worker thread exited.", _core);
  }

  void add_locked_value(const pool_t pool_id, Component::IKVStore::key_t key, void * target) {
    if(option_DEBUG)
      PLOG("shard: locked value (target=%p)", target);
    _locked_values[target] = std::make_pair(pool_id, key);
  }
  
  void release_locked_value(const void * target) {
    auto i = _locked_values.find(target);
    if(i == _locked_values.end())
      throw Logic_exception("bad target to unlock value");
    
    _i_kvstore->unlock(i->second.first,i->second.second);

    if(option_DEBUG)
      PLOG("unlocked value: %p", target);

    _locked_values.erase(i);
  }

  void initialize_components(const Program_options& po);
  
  void check_for_new_connections();
  void main_loop();

  void process_message_pool_request(Connection_handler* handler,
                                    Protocol::Message_pool_request* msg);
  
  void process_message_IO_request(Connection_handler* handler,
                                  Protocol::Message_IO_request* msg);


  /** 
   * Register memory with network transport for direct IO.  Cache in map.
   * 
   * @param handler Connection handler
   * @param target Pointer to start or region
   * @param target_len Region length in bytes
   * 
   * @return Memory region handle
   */
  Connection_base::memory_region_t ondemand_register(Connection_handler * handler,
                                                     const void * target,
                                                     size_t target_len)
  {
    Connection_base::memory_region_t region;
    auto entry = _registered.find(target);
    if(entry != _registered.end()) {
      region = entry->second;
      if(option_DEBUG)
        PLOG("region already registered %p len=%lu", target, target_len);
    }
    else {
      region = handler->register_memory(target, target_len);
      _registered[target] = region;
      if(option_DEBUG)
        PLOG("registering memory with fabric transport %p len=%lu", target, target_len);
    }
    return region;
  }
  
private:
  
  const std::string                   _data_dir;
  bool                                _thread_exit = false;
  bool                                _forced_exit;
  unsigned                            _core;
  std::thread                         _thread;
  size_t                              _max_message_size;
  Component::IKVStore *               _i_kvstore;
  std::vector<Connection_handler *>   _handlers;

  std::map<const void *, Connection_base::memory_region_t>              _registered;
  std::map<const void *, std::pair<pool_t, Component::IKVStore::key_t>> _locked_values;
};


} // namespace Dawn

#endif

#endif // __SHARD_HPP__
