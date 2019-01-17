#ifndef __DAWN_SHARD_H__
#define __DAWN_SHARD_H__

#ifdef __cplusplus

#include <api/cluster_itf.h>
#include <api/components.h>
#include <api/fabric_itf.h>
#include <api/kvstore_itf.h>
#include <common/cpu.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <map>
#include <thread>
#include "connection_handler.h"
#include "dawn_config.h"
#include "fabric_transport.h"
#include "pool_manager.h"
#include "types.h"

namespace Dawn
{
class Connection_handler;

/* Adapter point */
using Shard_transport = Fabric_transport;

class Shard : public Shard_transport {
 private:
  using buffer_t = Shard_transport::buffer_t;
  using pool_t = Component::IKVStore::pool_t;

  bool option_DEBUG;

 public:
  Shard(int core,
        unsigned int port,
        const std::string provider,
        const std::string device,
        const std::string net,
        const std::string backend,
        const std::string pci_addr,
        const std::string pm_path,
        unsigned debug_level,
        bool forced_exit)
      : Shard_transport(provider, net, port), _core(core),
        _forced_exit(forced_exit),
        _thread(&Shard::thread_entry, this, backend, pci_addr, debug_level) {
    Dawn::Global::debug_level = debug_level;
    option_DEBUG = Dawn::Global::debug_level > 1;
  }

  ~Shard() {
    _thread_exit = true;
    /* TODO: unblock */
    _thread.join();
    _i_kvstore->release_ref();
  }

  bool exited() const { return _thread_exit; }

 private:
  void thread_entry(const std::string& backend,
                    const std::string& pci_addr,
                    unsigned debug_level) {
    if (option_DEBUG) PLOG("shard:%u worker thread entered.", _core);

    if (set_cpu_affinity(1UL << _core) != 0)
      throw General_exception("unable to set cpu affinity (%lu)", _core);

    initialize_components(backend, pci_addr, debug_level);

    main_loop();

    if (option_DEBUG) PLOG("shard:%u worker thread exited.", _core);
  }

  void add_locked_value(const pool_t pool_id,
                        Component::IKVStore::key_t key,
                        void* target) {
    if (option_DEBUG) PLOG("shard: locked value (target=%p)", target);
    _locked_values[target] = std::make_pair(pool_id, key);
  }

  void release_locked_value(const void* target) {
    auto i = _locked_values.find(target);
    if (i == _locked_values.end())
      throw Logic_exception("bad target to unlock value");

    _i_kvstore->unlock(i->second.first, i->second.second);

    if (option_DEBUG) PLOG("unlocked value: %p", target);

    _locked_values.erase(i);
  }

  void initialize_components(const std::string& backend,
                             const std::string& pci_addr,
                             unsigned debug_level);

  void check_for_new_connections();

  void main_loop();

  void process_message_pool_request(Connection_handler* handler,
                                    Protocol::Message_pool_request* msg);

  void process_message_IO_request(Connection_handler* handler,
                                  Protocol::Message_IO_request* msg);

 private:
  bool _thread_exit = false;
  bool _forced_exit;
  unsigned _core;
  std::thread _thread;
  size_t _max_message_size;
  Component::IKVStore* _i_kvstore;
  std::vector<Connection_handler*> _handlers;

  std::map<const void*, std::pair<pool_t, Component::IKVStore::key_t>>
      _locked_values;
};

}  // namespace Dawn

#endif

#endif  // __SHARD_HPP__
