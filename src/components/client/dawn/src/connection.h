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
#ifndef __DAWN_CLIENT_HANDLER_H__
#define __DAWN_CLIENT_HANDLER_H__

#include <api/fabric_itf.h>
#include <api/dawn_itf.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <unistd.h>
#include <map>
#include <set>

#include "buffer_manager.h"
#include "dawn_client_config.h"

#include "client_fabric_transport.h"

/* Enable this to introduce locks to prevent re-entry by multiple
   threads.  The client is not re-entrant because of the state machine
   and multi-packet operations
*/
#define THREAD_SAFE_CLIENT

#ifdef THREAD_SAFE_CLIENT
#define API_LOCK() std::lock_guard<std::mutex> g(_api_lock);
#else
#define API_LOCK()
#endif

namespace Dawn
{
namespace Client
{
/* Adaptor point for other transports */
using Connection_base = Dawn::Client::Fabric_transport;

/**
 * Client side connection handler
 *
 */
class Connection_handler : public Connection_base {
  const bool option_DEBUG = Dawn::Global::debug_level > 1;

 public:
  using memory_region_t = typename Transport::memory_region_t;

  /**
   * Constructor
   *
   * @param connection
   *
   * @return
   */
  Connection_handler(Connection_base::Transport* connection);

  ~Connection_handler();
  

 private:
  enum State {
    INITIALIZE,
    HANDSHAKE_SEND,
    HANDSHAKE_GET_RESPONSE,
    SHUTDOWN,
    STOPPED,
    READY,
    POST,
    WAIT_RESPONSE,
  };

  State _state = State::INITIALIZE;

 public:
  using pool_t = uint64_t;

  void bootstrap()
  {
    set_state(INITIALIZE);
    while (tick() > 0)
      ;
  }

  void shutdown()
  {
    set_state(SHUTDOWN);
    while (tick() > 0) sleep(1);
  }

  pool_t open_pool(const std::string name,
                   uint32_t          flags);

  pool_t create_pool(const std::string name,
                     const size_t      size,
                     uint32_t          flags,
                     uint64_t          expected_obj_count);

  status_t close_pool(pool_t pool);

  status_t delete_pool(const std::string& name);

  status_t configure_pool(const Component::IKVStore::pool_t pool,
                          const std::string& json);

  status_t put(const pool_t      pool,
               const std::string key,
               const void*       value,
               const size_t      value_len,
               uint32_t          flags);

  status_t put(const pool_t pool,
               const void*  key,
               const size_t key_len,
               const void*  value,
               const size_t value_len,
               uint32_t flags);

  status_t put_direct(const pool_t                         pool,
                      const std::string&                   key,
                      const void*                          value,
                      const size_t                         value_len,
                      Component::IKVStore::memory_handle_t handle,
                      uint32_t                         flags);

  status_t get(const pool_t pool, const std::string& key, std::string& value);

  status_t get(const pool_t       pool,
               const std::string& key,
               void*&             value,
               size_t&            value_len);

  status_t get_direct(const pool_t                         pool,
                      const std::string&                   key,
                      void*                                value,
                      size_t&                              out_value_len,
                      Component::IKVStore::memory_handle_t handle = Component::IKVStore::HANDLE_NONE);

  status_t erase(const pool_t pool,
                 const std::string& key);

  uint64_t key_hash(const void* key, const size_t key_len);

  uint64_t auth_id() const { return ((uint64_t) this); }

  size_t count(const pool_t pool);

  status_t get_attribute(const Component::IKVStore::pool_t pool,
                         const Component::IKVStore::Attribute attr,
                         std::vector<uint64_t>& out_attr,
                         const std::string* key);

  status_t get_statistics(Component::IDawn::Shard_stats& out_stats);

  status_t find(const Component::IKVStore::pool_t pool,
                const std::string& key_expression,
                const offset_t offset,
                offset_t& out_matched_offset,
                std::string& out_matched_key);

  bool check_message_size(size_t size) const { return size > _max_message_size; }

 private:
  
  /**
   * FSM tick call
   *
   */
  int tick();

  /**
   * FSM state change
   *
   * @param s State to change to
   */
  inline void set_state(State s)
  {
    _state = s;
  } /* we could add transition checking later */

  /**
   * Put used when the value exceeds the size of the basic
   * IO buffer (e.g., 2MB).  This version of put will perform
   * a two-stage exchange, advance notice and then value
   *
   * @param pool Pool identifier
   * @param key Key
   * @param key_len Key length
   * @param value Value
   * @param value_len Value length
   *
   * @return
   */
  status_t two_stage_put_direct(const pool_t                         pool,
                                const void*                          key,
                                const size_t                         key_len,
                                const void*                          value,
                                const size_t                         value_len,
                                Component::IKVStore::memory_handle_t handle,
                                uint32_t                             flags);


 private:
#ifdef THREAD_SAFE_CLIENT
  std::mutex _api_lock;
#endif

  bool     _exit                = false;
  uint64_t _request_id          = 0;
  size_t   _max_message_size    = 0;
  size_t   _max_inject_size     = 0;

  struct {
    bool short_circuit_backend = false;
  } _options;
};

}  // namespace Client
}  // namespace Dawn

#endif
