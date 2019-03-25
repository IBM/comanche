/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef __DAWN_SHARD_H__
#define __DAWN_SHARD_H__

#ifdef __cplusplus

#include <api/cluster_itf.h>
#include <api/components.h>
#include <api/fabric_itf.h>
#include <api/kvstore_itf.h>
#include <api/kvindex_itf.h>
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
  using pool_t   = Component::IKVStore::pool_t;

  unsigned option_DEBUG;

 public:
  Shard(int               core,
        unsigned int      port,
        const std::string provider,
        const std::string device,
        const std::string net,
        const std::string backend,
        const std::string index,
        const std::string pci_addr,
        const std::string pm_path,
        const std::string dax_config,
        unsigned          debug_level,
        bool              forced_exit)
      : Shard_transport(provider, net, port), _core(core),
        _forced_exit(forced_exit), _thread(&Shard::thread_entry,
                                           this,
                                           backend,
                                           index,
                                           pci_addr,
                                           dax_config,
                                           debug_level)
  {
    option_DEBUG = Dawn::Global::debug_level = debug_level;
  }

  ~Shard()
  {
    _thread_exit = true;
    /* TODO: unblock */
    _thread.join();

    assert(_i_kvstore);
    _i_kvstore->release_ref();

    if (_index_factory)
      _index_factory->release_ref();
  }

  bool exited() const { return _thread_exit; }

 private:
  void thread_entry(const std::string& backend,
                    const std::string& index,
                    const std::string& pci_addr,
                    const std::string& dax_config,
                    unsigned           debug_level)
  {
    if (option_DEBUG > 2) PLOG("shard:%u worker thread entered.", _core);

    /* pin thread */
    cpu_mask_t mask;
    mask.add_core(_core);
    set_cpu_affinity_mask(mask);

    initialize_components(backend, index, pci_addr, dax_config, debug_level);

    main_loop();

    if (option_DEBUG > 2) PLOG("shard:%u worker thread exited.", _core);
  }

  void add_locked_value(const pool_t               pool_id,
                        Component::IKVStore::key_t key,
                        void*                      target)
  {
    if (option_DEBUG > 2) PLOG("shard: locked value (target=%p)", target);
    _locked_values[target] = std::make_pair(pool_id, key);
  }

  void release_locked_value(const void* target)
  {
    auto i = _locked_values.find(target);
    if (i == _locked_values.end())
      throw Logic_exception("bad target to unlock value");

    _i_kvstore->unlock(i->second.first, i->second.second);

    if (option_DEBUG > 2) PLOG("unlocked value: %p", target);

    _locked_values.erase(i);
  }

  void initialize_components(const std::string& backend,
                             const std::string& index,
                             const std::string& pci_addr,
                             const std::string& dax_config,
                             unsigned           debug_level);

  void check_for_new_connections();

  void main_loop();

  void process_message_pool_request(Connection_handler* handler,
                                    Protocol::Message_pool_request* msg);

  void process_message_IO_request(Connection_handler* handler,
                                  Protocol::Message_IO_request* msg);

 private:
  bool                             _thread_exit = false;
  bool                             _forced_exit;
  unsigned                         _core;
  std::thread                      _thread;
  size_t                           _max_message_size;
  Component::IKVStore*             _i_kvstore;
  Component::IKVIndex_factory*     _index_factory;
  std::vector<Connection_handler*> _handlers;

  std::map<const void*, std::pair<pool_t, Component::IKVStore::key_t>>
      _locked_values;
};

}  // namespace Dawn

#endif

#endif  // __SHARD_HPP__
