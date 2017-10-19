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

#ifndef __COMANCHE_AGENT_H__
#define __COMANCHE_AGENT_H__

#include <mutex>
#include <string>
#include <map>
#include <thread>
#include <stack>

#include "channel.h"
#include "messaging.h"
#include "json_config.h"

class Port_list
{
public:
  Port_list() {
    for(unsigned p=15800;p<15900;p++)
      _available_ports.push_back(p);        
  }

  int assign_port() {
    std::lock_guard<std::mutex> g(_lock);
    if(_available_ports.empty())
      throw General_exception("out of available port numbers");
    int p = _available_ports.back();
    _available_ports.pop_back();
    return p;
  }

  void release_port(int port) {
    std::lock_guard<std::mutex> g(_lock);
    _available_ports.push_back(port);
  }    

private:
  std::vector<int> _available_ports;
  std::mutex       _lock;
};
  

/** 
 * Base class common to both Volume and Storage agents.
 * 
 */
class Agent
{
  friend class Volume_agent_session;
  
public:
  void wait_for_event()
  {
    _mgroup->wait_for_ingress_event();
  }

protected:

  enum {
    NODE_STATE_ACTIVE = 1,
    NODE_STATE_LEFT   = 2,
    NODE_STATE_MIA    = 3,
  };

  /** 
   * Constructor
   * 
   * @param configuration_file Configuration file name or NULL
   */
  Agent(const char * configuration_file) :
    _vol_config(configuration_file),
    _fsm_thread_exit(false)
  {
  }

  virtual ~Agent()
  {
  }

  /** 
   * Get current host member id (hostname)
   * 
   * 
   * @return Member identify string. No release needed.
   */
  static const char* member_id()
  {
    static char hostname[HOST_NAME_MAX];
    check_zero(gethostname(hostname, HOST_NAME_MAX));
    return hostname;
  }

  void messaging_callback(int type, const char *arg0, const char *arg1, const char *arg2);

  /** 
   * Establish data plane connection with peer
   * 
   * @param peer_name 
   * @param channel 
   * 
   * @return 
   */
  status_t connect_peer(const char * peer_name, Channel& channel);
  
  /** 
   * Lookup UUID in local map or send WHOIS request
   * 
   * @param peer 
   * 
   * @return 
   */
  std::string lookup_uuid(const char * peer)
  {
    int retries = 3;
    while(_uuid_map[peer].empty() && retries > 0) {
      _mgroup->send_who_is(peer);
      /* wait for a ingress message event */
      if(_mgroup->wait_for_ingress_event() == E_RECV_TIMEOUT) {
        PLOG("peer: %s not responding",peer);
      }
      retries--;
    }
    return _uuid_map[peer];
  }

  inline Json_configuration& config() { return _vol_config; }

  enum {
    ROLE_LEADER    = 0x1,
    ROLE_SINGLETON = 0x2,    
  };
  
  Message_group *            _mgroup;
  Json_configuration         _vol_config;
  std::string                _volume_id;
  unsigned long              _role;
  std::thread*               _fsm_thread;
  bool                       _fsm_thread_exit;

  std::mutex                 _state_lock;
  std::map<std::string, int> _active_nodes;
  std::map<std::string, std::string> _uuid_map;
  std::mutex                 _fsm_postbox_lock;
  std::stack<int>            _fsm_postbox;

  static Port_list           _port_list;
  
};



#endif // __COMANCHE_AGENT_H__
