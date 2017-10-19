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

#include <arpa/inet.h>
#include <sys/socket.h>
#include <list>
#include <netdb.h>
#include <ifaddrs.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/if_link.h>
#include <unistd.h>
#include <common/utils.h>
#include <common/cycles.h>
#include <rte_malloc.h>

#include <eal_init.h>
#include "volume_agent.h"
#include "volume_agent_session.h"
#include "json_config.h"
#include "protocols.h"



void Volume_agent::__va_callback_trampoline(void *cbarg,
                                                   int type,
                                                   const char *arg0,
                                                   const char *arg1,
                                                   const char *arg2)
{
  (static_cast<Volume_agent *>(cbarg))->messaging_callback(type,arg0,arg1,arg2);
}

/** 
 * Volume_agent class
 * 
 */

Volume_agent::Volume_agent(const char * configuration_file) :
  Agent(configuration_file)
{
  DPDK::eal_init(64); /* 64 MB memory limit */

  /* create message group */
  _mgroup = new Message_group(member_id(),
                              _vol_config.volume_id(), /* use volume id as group name */
                              __va_callback_trampoline,
                              this);

}

Volume_agent::~Volume_agent()
{
  TRACE();
  for(auto& s: _sessions)
    delete s;
  
  delete _mgroup; /* this will cleanly remove us from the messaging group */
}


Volume_agent_session * Volume_agent::create_session(unsigned core)
{
  // Volume_agent_session * session = new Volume_agent_session(*this, _vol_config.initial_leader(), core);
  // _sessions.push_back(session);
  //   return session;
    
  return (new Volume_agent_session(*this, _vol_config.initial_leader(), core));
}

void Volume_agent::messaging_callback(int type,
                                      const char *arg0,
                                      const char *arg1,
                                      const char * arg2)
{
  PLOG("inbound message event! type=%d", type);

  switch(type) {
  case Message_group::EVENT_MEMBER_WHISPER:
    {
      PLOG("node %s whispered : %s %s", arg0, arg1, arg2);
    }
  }

  // switch(type) {
  // case Message_group::EVENT_MEMBER_SHOUTED:
  //   {
  //     std::lock_guard<std::mutex> lock(_state_lock);
  //     PLOG("node %s shouted : %s %s", arg0, arg1, arg2);
  //     break;
  //   }
  // }

  Agent::messaging_callback(type,arg0,arg1,arg2); /* call base class */
}

  



// global statics
Port_list Agent::_port_list;


