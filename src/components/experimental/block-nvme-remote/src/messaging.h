/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __COMANCHE_MESSAGING_H__
#define __COMANCHE_MESSAGING_H__

#include <common/exceptions.h>
#include <common/logging.h>
#include <common/utils.h>
#include <common/semaphore.h>
#include <czmq.h>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <stdint.h>
#include <string>
#include <vector>
#include <zyre.h>

/**
 * Reference: https://github.com/zeromq/zyre
 *
 */

/**
 * A message group is a collection of communicating peers.  This class
 * provides the basic messaging layer for the volume group members.
 *
 */
class Message_group {
  static constexpr int OPTION_evasive_timeout = 3000; // ms
  static constexpr bool OPTION_verbose = false;

public:
  enum {
    EVENT_MEMBER_SHOUTED = 1,
    EVENT_MEMBER_WHISPER = 2,
    EVENT_MEMBER_EVASIVE = 3,
    EVENT_MEMBER_LEFT    = 4,
    EVENT_MEMBER_JOINED  = 5,
  };

  /**
   * Constructor
   *
   * @param member_id Unique id for this member
   * @param group_name Unique name of group
   * @param callback Callback for events and results
   * @param callback_arg 
   */
  Message_group(const char* member_id,
                const char* group_name,
                std::function<void(void*, int, const char *, const char *, const char*)> callback,
                void * callback_arg)
      : _initialized(false),
        _group_name(group_name),
        _member_id(member_id),
        _callback(callback),
        _callback_arg(callback_arg),
        _z_node(NULL)
  {
    assert(callback);
    assert(group_name);

    _z_actor = zactor_new(__actor_trampoline, this);
    assert(_z_actor);
    while (!_initialized) {
      usleep(100000);
      PLOG("message_group: waiting for actor..");
    }
    PLOG("message group: init OK. _z_node=%p",_z_node);
    assert(_z_node);

  }

  /**
   * Send a launch command to a peer
   *
   * @param peer Target peer
   * @param remote_cmd Remote command
   *
   * @return S_OK on success
   */
  status_t send_launch(const char *remote_cmd) {
    if (zstr_sendx(_z_actor, "$LAUNCH", remote_cmd, NULL))
      throw General_exception("Message_group send_launch failed unexpectedly");

    char *status = zmsg_popstr(zactor_recv(_z_actor));
    if (status == NULL)
      return E_FAIL;
    return streq(status, "OK!") ? S_OK : E_FAIL;
  }

  /**
   * Kill a remote process
   *
   * @param pattern Pkill pattern (man pkill)
   *
   * @return S_OK on success
   */
  status_t send_pkill(const char *pattern) {
    if (zstr_sendx(_z_actor, "$PKILL", pattern, NULL))
      throw General_exception("Message_group send_pkill failed unexpectedly.");

    char *status = zmsg_popstr(zactor_recv(_z_actor));
    if (status == NULL)
      return E_FAIL;
    return streq(status, "OK!") ? S_OK : E_FAIL;
  }

  /**
   * Send message to all members of the group
   *
   * @param msg Message to send
   */
  void broadcast(const char *msg) {
    assert(_z_actor);
    assert(_initialized);
    assert(msg);
    // synchronous: see https://github.com/zeromq/czmq/blob/master/src/zstr.c
    int rc = zstr_sendx(_z_actor, "$SHOUT", msg, NULL);
    if (rc)
      throw General_exception("zstr_sendx failed unexpectedly. rc=%d", rc);
  }

  /** 
   * Wait on some message to arrive and trigger an event.
   * 
   */
  status_t wait_for_ingress_event() {
    Common::Semaphore s;
    {
      std::unique_lock<std::mutex> lk(_waiting_threads_lock);
      _waiting_threads.push_back(&s);
    }
    if(s.wait_for(500)) {
      return S_OK;
    }
    else {
      return E_RECV_TIMEOUT;
    }
  }  

  /** 
   * Asynchronously send a ?WHOIS command to resolve peer name to UUID
   * 
   * @param peer_name Peer name 
   */
  void send_who_is(const char* peer_name) {
    assert(peer_name);
    assert(_z_actor);
    int rc = zstr_sendx(_z_actor, "?WHOIS", peer_name, NULL);
    if (rc)
      throw General_exception("zstr_sendx failed unexpectedly. rc=%d", rc);    
  }

  /** 
   * Get the local node's UUID
   * 
   * 
   * @return 
   */
  const char * our_uuid() {
    assert(_z_node);
    return zyre_uuid(_z_node);
  }

  /** 
   * Broadcast a two part message (command, arg)
   * 
   * @param cmd Command string
   * @param arg Argument string
   */
  void broadcast(const char *cmd, const char *arg) {
    assert(_z_actor);
    assert(_initialized);
    assert(cmd);
    assert(arg);
    
    // synchronous: see https://github.com/zeromq/czmq/blob/master/src/zstr.c
    int rc = zstr_sendx(_z_actor, "$SHOUT", cmd, arg, NULL);
    if (rc)
      throw General_exception("zstr_sendx failed unexpectedly. rc=%d", rc);
  }

  /**
   * Send a message to a specific peer
   *
   * @param peer Peer UUID
   * @param msg Message to send
   */
  void whisper(const char *peer, const char *msg) {
    int rc = zstr_sendx(_z_actor, "$WHISPER", peer, msg, NULL);
    if (rc)
      throw General_exception("Message_group whisper failed unexpectedly. rc=%d", rc);
  }

  /** 
   * Send two part message to a specific peer
   * 
   * @param peer Peer UUID
   * @param cmd Command
   * @param arg Argument
   */
  void whisper(const char* peer, const char *cmd, const char *arg) {
    assert(_z_actor);
    assert(_initialized);
    assert(cmd);
    assert(arg);
    
    // synchronous: see https://github.com/zeromq/czmq/blob/master/src/zstr.c
    int rc = zstr_sendx(_z_actor, "$WHISPER", peer, cmd, arg, NULL);
    if (rc)
      throw General_exception("zstr_sendx failed unexpectedly. rc=%d", rc);
  }
  
  /**
   * Destructor
   *
   */
  virtual ~Message_group() { zactor_destroy(&_z_actor); }

private:
  static void __actor_trampoline(zsock_t *pipe, void *args) {
    static_cast<Message_group *>(args)->actor_entry(pipe);
  }

  void actor_entry(zsock_t *pipe) {
    PLOG("actor_entry");
    assert(_initialized == false);

    /* verify zyre version */
#ifndef NDEBUG
    uint64_t version = zyre_version();
    assert((version / 10000) % 100 == ZYRE_VERSION_MAJOR);
    assert((version / 100) % 100 == ZYRE_VERSION_MINOR);
    assert(version % 100 == ZYRE_VERSION_PATCH);
#endif
    _z_node = zyre_new(_member_id.c_str());
    assert(_z_node);
    
    /* at the moment we use UDP beaconing -- for wider area networks
       UDP beasconing may need to be replaced with a gossip protocol.
       This would require making the gossip endpoint reliable.
    */
#ifdef USE_GOSSIP
    zyre_set_endpoint(_z_node, "tcp://*:8888");      
    zyre_gossip_bind(_z_node,"inproc://gossip-hub");
#endif
    zyre_print(_z_node);

    if (OPTION_verbose)
      zyre_set_verbose(_z_node);

    zyre_start(_z_node);

    /* join group */
    check_zero(zyre_join(_z_node, _group_name.c_str()));

    zsock_signal(pipe, 0);
    bool exit = false;
    zpoller_t *poller = zpoller_new(pipe, zyre_socket(_z_node), NULL);

    zyre_set_evasive_timeout(_z_node, OPTION_evasive_timeout);

    _initialized = true;
    while (!exit) {
      
      void *which = zpoller_wait(poller, -1);
      if (which == pipe) {
        
        zmsg_t *msg = zmsg_recv(which);
        if (!msg) {
          PLOG("message interrupt!");
          break; //  Interrupted
        }

        char *command = zmsg_popstr(msg);
        
        if (streq(command, "$TERM"))
          exit = true;
        else {
          if (streq(command, "$SHOUT")) {
            zyre_shout(_z_node, _group_name.c_str(), &msg);
          }
          else if (streq(command, "$WHISPER")) {
            char *peer = zmsg_popstr(msg);
            assert(peer);
            zyre_whisper(_z_node, peer, &msg);
          }
          else if (streq(command, "$LAUNCH")) {
            char *cmd = zmsg_popstr(msg);
            /* broadcast launch command */
            zyre_shouts(_z_node, _group_name.c_str(), "$REMOTE_LAUNCH:%s", cmd);

            free(cmd);
            zstr_sendx(which, "OK!", NULL);
          }
          else if (streq(command, "?WHOIS")) {
            PLOG("broadcasting ?WHOIS");
            zmsg_dump(msg);
            zmsg_pushstr(msg, "?WHOIS");
            zyre_shout(_z_node, _group_name.c_str(), &msg);
          }
          else if (streq(command, "$PKILL")) {
            char *cmd = zmsg_popstr(msg);
            /* broadcast launch command */
            zyre_shouts(_z_node, _group_name.c_str(), "$PKILL:%s", cmd);
            free(cmd);
            zstr_sendx(which, "OK!", NULL);
          }
          else {
            throw General_exception("zyre: invalid message to actor (%s)",
                                    command);
          }
        }
        free(command);
        zmsg_destroy(&msg);

      }
      else if (which == zyre_socket(_z_node)) {

        zmsg_t *msg = zmsg_recv(which);
        char *event = zmsg_popstr(msg);
        char *peer = zmsg_popstr(msg);
        char *name = zmsg_popstr(msg);
        char *message0 = zmsg_popstr(msg);
        char *message1 = zmsg_popstr(msg);
        char *message2 = zmsg_popstr(msg);

        assert(_callback);

        if (streq(event, "ENTER")) {
          _callback(_callback_arg, EVENT_MEMBER_JOINED, name, NULL, NULL);
        }
        else if (streq(event, "EXIT")) {
          _callback(_callback_arg, EVENT_MEMBER_LEFT, name, NULL, NULL);
        }
        else if (streq(event, "SHOUT")) {
          _callback(_callback_arg, EVENT_MEMBER_SHOUTED, name, message1, message2);
        }
        else if (streq(event, "EVASIVE")) {
          _callback(_callback_arg, EVENT_MEMBER_EVASIVE, name, NULL, NULL);
        }
        else if (streq(event, "WHISPER")) {
          _callback(_callback_arg, EVENT_MEMBER_WHISPER, name, message0, message1);
        }

        free(event);
        free(peer);
        free(name);
        free(message0);
        free(message1);
        free(message2);
        zmsg_destroy(&msg);

        /* signal event */
        {
          std::unique_lock<std::mutex> lk(_waiting_threads_lock);
          while(!_waiting_threads.empty()) {
            auto m = _waiting_threads.back();
            m->post();
            _waiting_threads.pop_back();
          }
        }
      }

    }

    zpoller_destroy(&poller);
    zyre_stop(_z_node);
    zclock_sleep(100);
    zyre_destroy(&_z_node);

    PLOG("actor: exit");
  }

private:
  bool _initialized;
  std::string _group_name;
  std::string _member_id;
  
  std::function<void(void *,int, const char *, const char *, const char*)> _callback;
  void* _callback_arg;
  
  zyre_t *_z_node;
  zactor_t *_z_actor;

  std::mutex                       _waiting_threads_lock;
  std::vector<Common::Semaphore *> _waiting_threads;

};


#endif
