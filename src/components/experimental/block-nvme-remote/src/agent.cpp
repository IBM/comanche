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

#include "agent.h"
#include "messaging.h"


void Agent::messaging_callback(int type, const char *arg0, const char *arg1, const char *arg2)
{
  switch(type)
    {
    case Message_group::EVENT_MEMBER_SHOUTED:
      {
        if(streq(arg1,"?WHOIS")) {
          PLOG("received ?WHOIS: %s from %s", arg2, arg0);
          if(streq(arg2,member_id())) {
            /* we could unicast the reply and how the ?WHOIS send their UUID */
            _mgroup->broadcast("$I_AM",_mgroup->our_uuid());
          }
        }
        else if(streq(arg1, "$I_AM")) {
          std::lock_guard<std::mutex> lock(_state_lock);
          //          PLOG("processing I_AM: %s is uuid:%s", arg0, arg2);
          _uuid_map[arg0] = arg2;
        }
        else {
          PLOG("ignoring shout: %s %s %s", arg0, arg1, arg2);
        }
        break;
      }
    case Message_group::EVENT_MEMBER_JOINED:
      {
        std::lock_guard<std::mutex> lock(_state_lock);
        assert(arg0);
        PLOG("node (%s) joined", arg0);
        _active_nodes[arg0] = NODE_STATE_ACTIVE;
        break;
      }
    case Message_group::EVENT_MEMBER_LEFT:
      {
        std::lock_guard<std::mutex> lock(_state_lock);
        assert(arg0);
        PLOG("node %s left", arg0);
        _active_nodes[std::string(arg0)] = NODE_STATE_LEFT;
        /* post event to fsm thread */
        {
          std::lock_guard<std::mutex> g(_fsm_postbox_lock);
          _fsm_postbox.push(type);
        }

        break;
      }
    case Message_group::EVENT_MEMBER_EVASIVE:
      {
        std::lock_guard<std::mutex> lock(_state_lock);
        assert(arg0);
        PLOG("node %s is MIA", arg0);
        _active_nodes[std::string(arg0)] = NODE_STATE_MIA;
        /* post event to fsm thread */
        {
          std::lock_guard<std::mutex> g(_fsm_postbox_lock);
          _fsm_postbox.push(type);
        }
      
        break;
      }
    }
}

status_t Agent::connect_peer(const char * peer_name, Channel& channel)
{
  /* send (whisper) configuration to peer */
  std::string uuid;
  uuid = lookup_uuid(peer_name);

  int port = _port_list.assign_port();

  if(uuid.empty())
    return E_NO_RESPONSE;

  std::stringstream ss;
  // ss << "{\"volume_info\": " << _vol_config.config_string()
  //    << ", \"connection_info\" : " << Channel::get_connection_info(port) << "}";
  ss << Channel::get_connection_info(port);
  PLOG("sending config (%s) to %s", ss.str().c_str(), uuid.c_str());
  _mgroup->whisper(uuid.c_str(), "$CONNECT", ss.str().c_str() );
  

  /* wait for connection */
  PLOG("waiting for channel connection");

  status_t rc = channel.connect(peer_name, NULL, port);
  //  _port_list.release_port(port);

  return rc;
}

