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
#ifndef __SHARD_CONNECTION_H__
#define __SHARD_CONNECTION_H__

#ifdef __cplusplus

#include <api/components.h>
#include <api/kvstore_itf.h>
#include <common/cpu.h>
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/cycles.h>
#include <sys/mman.h>
#include <map>
#include <queue>
#include <set>
#include <thread>

#include "buffer_manager.h"
#include "dawn_config.h"
#include "pool_manager.h"
#include "protocol.h"
#include "region_manager.h"

#include <api/fabric_itf.h>
#include "fabric_connection_base.h"  // default to fabric transport

namespace Dawn
{
using Connection_base = Fabric_connection_base;

class Connection_handler
    : public Connection_base
    , public Region_manager
    , public Pool_manager {
 private:

  unsigned option_DEBUG = Dawn::Global::debug_level;
  static constexpr uint64_t STALL_TICKS = 20; /*< number of ticks to stall after post */
  
  /* Adaptor point for different transports */
  using Connection = Component::IFabric_server;
  using Factory    = Component::IFabric_server_factory;

 public:
  enum {
    TICK_RESPONSE_CONTINUE        = 0,
    TICK_RESPONSE_BOOTSTRAP_SPAWN = 1,
    TICK_RESPONSE_CLOSE           = 0xFF,
  };

  enum {
    ACTION_NONE = 0,
    ACTION_RELEASE_VALUE_LOCK,
  };

 protected:
  enum State {
    INITIALIZE,
    POST_HANDSHAKE,
    WAIT_HANDSHAKE,
    WAIT_HANDSHAKE_RESPONSE_COMPLETION,
    POST_MAX_RECVS,
    POST_MSG_RECV,
    WAIT_NEW_MSG_RECV,
    WAIT_RECV_VALUE,
  };

  State _state = State::INITIALIZE;

 public:
  Connection_handler(Factory* factory, Connection* connection)
      : Connection_base(factory, connection), Region_manager(connection)
  {
    _pending_actions.reserve(Buffer_manager<Connection>::DEFAULT_BUFFER_COUNT);
    _pending_msgs.reserve(Buffer_manager<Connection>::DEFAULT_BUFFER_COUNT);
    _freq_mhz = Common::get_rdtsc_frequency_mhz();
  }

  ~Connection_handler()
  {
    dump_stats();
    //    exit(0); /* for profiler */
  }

  /**
   * State machine transition tick.  It is really important that this tick
   * execution duration is small, so that other connections are not impacted by
   * the thread not returning to them.
   *
   */
  int tick();

  /**
   * Change state in FSM
   *
   * @param s
   */
  inline void set_state(State s)
  {
    _state = s; /* we could add transition checking later */
  }


  /**
   * Check for network completions
   *
   */
  Fabric_connection_base::Completion_state check_network_completions()    
  {
    auto state = poll_completions();
    if ( state == Fabric_connection_base::Completion_state::ADDED_DEFERRED_LOCK ) {
      /* deferred unlocks */
      if (_deferred_unlock) {
        if (option_DEBUG > 2)
          PLOG("adding action for deferred unlocking value @ %p",
               _deferred_unlock);
        add_pending_action(action_t{ACTION_RELEASE_VALUE_LOCK, _deferred_unlock});
        _deferred_unlock = nullptr;
      }
    }
    return state;
  }

  /**
   * Get pending message from the connection
   *
   * @param msg [out] Pointer to base protocol message
   *
   * @return Pointer to buffer holding the message or null if there are none
   */
  inline buffer_t* get_pending_msg(Dawn::Protocol::Message*& msg)
  {
    if (_pending_msgs.empty()) return nullptr;
    auto iob = _pending_msgs.back();
    assert(iob);
    _pending_msgs.pop_back();
    msg = static_cast<Dawn::Protocol::Message*>(iob->base());
    return iob;
  }

  /**
   * Get deferrd action
   *
   * @param action [out] Action
   *
   * @return True if action
   */
  inline bool get_pending_action(action_t& action)
  {
    if (_pending_actions.empty()) return false;
    action = _pending_actions.back();
    _pending_actions.pop_back();
    return true;
  }

  /**
   * Add an action to the pending queue
   *
   * @param action Action to add
   */
  inline void add_pending_action(const action_t action)
  {
    _pending_actions.push_back(action);
  }

  /**
   * Post a response
   *
   * @param iob IO buffer to post
   */
  inline void post_response(buffer_t* iob, buffer_t* val_iob = nullptr)
  {
    assert(iob);

    post_send_buffer(iob, val_iob);
    if (val_iob) {
      _posted_value_buffer = val_iob;
    }

    _stats.response_count++;

    set_state(POST_MSG_RECV); /* don't wait for this, let it be picked up in
                                 the check_completions cycle */
  }

  /**
   * Set up for pending value send/recv
   *
   * @param target
   * @param target_len
   * @param region
   */
  void set_pending_value(void*  target,
                         size_t target_len,
                         Component::IFabric_connection::memory_region_t region);

  inline uint64_t auth_id() const { return (uint64_t) this; /* temp */ }

  inline size_t max_message_size() const { return _max_message_size; }

  inline uint64_t stall_tick() {
    if(_stall_tick == 0) return 0;
    _stall_tick--;
    return _stall_tick;
  }

  inline void stall() {
    _stall_tick = STALL_TICKS;
  }
  
 private:
  struct {
    uint64_t response_count               = 0;
    uint64_t recv_msg_count               = 0;
    uint64_t wait_recv_value_misses       = 0;
    uint64_t wait_msg_recv_misses         = 0;
    uint64_t wait_respond_complete_misses = 0;
    uint64_t last_count                   = 0;
    uint64_t next_stamp                   = 0;
  } _stats __attribute__((aligned(8)));

  void dump_stats()
  {
    PINF("-----------------------------------------");
    PINF("| Connection Handler Statistics         |");
    PINF("-----------------------------------------");
    PINF("Ticks                       : %lu", _tick_count);
    PINF("NEW_MSG_RECV misses         : %lu", _stats.wait_msg_recv_misses);
    PINF("Recv message count          : %lu", _stats.recv_msg_count);
    PINF("Response count              : %lu", _stats.response_count);
    PINF("WAIT_RECV_VALUE misses      : %lu", _stats.wait_recv_value_misses);
    PINF("WAIT_RESPOND_COMPLETE misses: %lu", _stats.wait_respond_complete_misses);
    PINF("-----------------------------------------");
  }

 private:
  uint64_t               _tick_count __attribute((aligned(8))) = 0;
  uint64_t               _stall_tick __attribute((aligned(8))) = 0;  
  std::vector<buffer_t*> _pending_msgs;
  std::vector<action_t>  _pending_actions;
  float                  _freq_mhz;
};

}  // namespace Dawn
#endif

#endif  // __SHARD_HPP__
