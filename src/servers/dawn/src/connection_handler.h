#ifndef __SHARD_CONNECTION_H__
#define __SHARD_CONNECTION_H__

#ifdef __cplusplus

#include <sys/mman.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <common/logging.h>
#include <common/exceptions.h>
#include <common/cpu.h>
#include <queue>
#include <thread>
#include <set>
#include <map>

#include "protocol.h"
#include "buffer_manager.h"
#include "dawn_config.h"

#include <api/fabric_itf.h>
#include "fabric_connection_base.h" // default to fabric transport

namespace Dawn {

  /* Adapter point for different transports */
  using Connection_base = Fabric_connection_base;


  class Connection_handler : public Connection_base
  {
  private:
    bool option_DEBUG = Dawn::Global::debug_level > 1;

    /* Adaptor point for different transports */
    using Connection = Component::IFabric_server;
    using Factory = Component::IFabric_server_factory;
  
  public:

    enum {
      TICK_RESPONSE_CONTINUE = 0,
      TICK_RESPONSE_CLOSE = 0xFF,
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
      NEW_MSG_RECV,
      POST_RECV_VALUE,
      WAIT_RECV_VALUE,
      //    WAIT_POST_RESPONSE,
    };

    State _state = State::INITIALIZE;

  public:

    Connection_handler(Factory * factory,
                       Connection * connection) : 
      Connection_base(factory, connection) {
      PLOG("Connection_handler: %p", this);
      _pending_actions.reserve(Buffer_manager<Connection>::DEFAULT_BUFFER_COUNT);
      _pending_msgs.reserve(Buffer_manager<Connection>::DEFAULT_BUFFER_COUNT);
    }

    ~Connection_handler() {
      dump_stats();
    }
  
    /**
     * State machine transition tick.  It is really important that this tick execution
     * duration is small, so that other connections are not impacted by the thread
     * not returning to them.
     *
     */
    int tick();

    /** 
     * Change state in FSM
     * 
     * @param s 
     */
    inline void set_state(State s) {
      _tick_count++;
      _state = s; /* we could add transition checking later */
    }

    /** 
     * Get pending message from the connection
     * 
     * @param msg [out] Pointer to base protocol message
     * 
     * @return Pointer to buffer holding the message or null if there are none
     */
    buffer_t * get_pending_msg(Dawn::Protocol::Message*& msg) {
      if(_pending_msgs.empty()) return nullptr;
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
    bool get_pending_action(action_t& action) {
      if(_pending_actions.empty()) return false;
      action = _pending_actions.back();
      _pending_actions.pop_back();
      return true;
    }

    /** 
     * Add an action to the pending queue
     * 
     * @param action Action to add
     */
    inline void add_pending_action(const action_t action) {
      _pending_actions.push_back(action);
    }

    /** 
     * Post a response
     * 
     * @param iob IO buffer to post
     */
    void post_response(buffer_t * iob, buffer_t * val_iob = nullptr) {
      assert(iob);

      post_send_buffer(iob, val_iob);
      if(val_iob) {
        _posted_value_buffer = val_iob;
      }
    
      _stats.response_count++;

      set_state(POST_MSG_RECV); /* don't wait for this, let it be picked up in
                                //the check_completions cycle */
      //set_state(WAIT_POST_RESPONSE);
    }

    /** 
     * Record pool as open
     * 
     * @param pool Pool identifier
     */
    void add_as_open_pool(pool_t pool, std::vector<::iovec>* regions = nullptr) {
      if(_open_pools.count(pool) > 0)
        throw Logic_exception("add_as_open_pool: pool already exists");
    
      _open_pools.insert(pool);

      /* if the backend store supports region exposure, then
         we simply register upfront 
      */
      if(regions) {
        for(auto& r : *regions) {
          assert(r.iov_base);
          assert(r.iov_len);
          auto h = register_memory(r.iov_base, r.iov_len);
          _pool_memory_handles[pool].push_back(h);
          PLOG("connection_handler: registered pool memory (%p,%lu) (%p)", r.iov_base, r.iov_len, h);
        }
      }	
    }

    /**
     * Get registered memory region if it exists
     */
    Connection_base::memory_region_t get_preregistered(pool_t pool) {
      auto i = _pool_memory_handles.find(pool);
      if(i == _pool_memory_handles.end()) return nullptr;
      assert(i->second.size() == 1);
      return i->second[0]; /* for the moment return the first region handle */
    }      
  
    /** 
     * Remove pool open record
     * 
     * @param pool Pool identifier
     */
    void remove_as_open_pool(pool_t pool) {
      _open_pools.erase(pool);
    }

    /** 
     * Set up for pending value send/recv
     * 
     * @param target 
     * @param target_len 
     * @param region 
     */
    void set_pending_value(void * target,
                           size_t target_len,
                           memory_region_t region);

    inline uint64_t auth_id() const { return (uint64_t) this; /* temp */ }

    inline const std::set<pool_t>& open_pool_set() { return _open_pools; }

    inline bool validate_pool(pool_t pool) const { return _open_pools.count(pool) == 1; }

    inline size_t max_message_size() const { return _max_message_size; }

  private:
    struct {
      unsigned long response_count = 0;
      unsigned long recv_msg_count = 0;
      unsigned long wait_recv_value_misses = 0;
      unsigned long wait_msg_recv_misses = 0;
      unsigned long wait_respond_complete_misses = 0;
    } _stats __attribute__((aligned(8)));


    void dump_stats() {
      PINF("-----------------------------------------");
      PINF("| Connection Handler Statistics         |");
      PINF("-----------------------------------------");
      PINF("Recv message count          : %lu", _stats.recv_msg_count);
      PINF("Response count              : %lu", _stats.response_count);     
      PINF("WAIT_MSG_RECV misses        : %lu", _stats.wait_msg_recv_misses);
      PINF("WAIT_RECV_VALUE misses      : %lu", _stats.wait_recv_value_misses);
      PINF("WAIT_RESPOND_COMPLETE misses: %lu", _stats.wait_respond_complete_misses);
      PINF("-----------------------------------------");
    }
  
  private:

    bool                   _complete;
    unsigned               _tick_count = 0;
    std::vector<buffer_t*> _pending_msgs;
    std::vector<action_t>  _pending_actions;
    int                    _response;
  
    std::set<pool_t> _open_pools; /*< pools currently open by client */
    std::map<pool_t, std::vector<Connection_base::memory_region_t>> _pool_memory_handles;
  };


} // namespace Dawn
#endif

#endif // __SHARD_HPP__
