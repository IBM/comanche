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
#include <pthread.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/if_link.h>
#include <unistd.h>
#include <algorithm>
#include <mutex>
#include <stack>
#include <common/utils.h>
#include <eal_init.h>

#include <rte_config.h>
#include <rte_malloc.h>
#include <rte_mempool.h>

/* from comanche */
#include <nvme_device.h>
#include <nvme_queue.h>

#include "storage_agent.h"
#include "storage_agent_session.h"
#include "protocols.h"
#include "dd-config.h"
#include "operations.h"
#include "json_config.h"
#include "version.h"


static void __sa_callback_trampoline(void *cbarg,
                                     int type,
                                     const char *arg0,
                                     const char *arg1,
                                     const char *arg2)
{
  (static_cast<Storage_agent *>(cbarg))->messaging_callback(type,arg0,arg1,arg2);
}


Storage_agent::Storage_agent(const char * configuration_file) :
  Agent(configuration_file)
{
  PINF("Storage_agent: Comanche v%s", COMANCHE_VERSION);
  DPDK::eal_init(256); /* MB memory limit */

  /* attached local SPDK devices - must be before messaging is activated! */
  attach_devices();
  
  _mgroup = new Message_group(member_id(),
                              _vol_config.volume_id(), /* use volume id as group name */
                              __sa_callback_trampoline,
                              this);

  /* get session cores */
  rapidjson::Value& cores = config().session_cores(member_id());
  for (rapidjson::SizeType i = 0; i < cores.Size(); i++) {
    _available_session_cores.push_back(cores[i].GetInt());
  }
  
}

Storage_agent::~Storage_agent()
{
  /* clean up sessions */
  for(auto& s: _sessions) {
    delete s;
  }
  
  delete _mgroup;
}

 

void Storage_agent::messaging_callback(int type,
                                       const char *arg0,
                                       const char *arg1,
                                       const char *arg2)
{
  Agent::messaging_callback(type,arg0,arg1,arg2); /* base class callback */
  
  switch(type) {    
  case Message_group::EVENT_MEMBER_SHOUTED:
    {
      std::lock_guard<std::mutex> lock(_state_lock);
      // if(streq(arg1, "$CONNECT")) {
      //    PNOTICE("Connect request from node %s recvd : %s", arg0, arg2);
      //    receive_configuration(arg2);
      //  }
      break;
    }
  case Message_group::EVENT_MEMBER_WHISPER:
    {
      std::lock_guard<std::mutex> lock(_state_lock);
      if(streq(arg1, "$CONNECT")) {
        PNOTICE("Connect request from node %s recvd : %s", arg0, arg2);
        // PLOG("node %s sent configuration : %s", arg0,arg2);
        // receive_configuration(arg2);

        std::stringstream ss(arg2);
        rapidjson::IStreamWrapper isw(ss);
        json_document_t doc;
        doc.ParseStream(isw);
        assert(doc["port"].IsInt());
        int port = doc["port"].GetInt();
        assert(port > 0);

        if(_storage_devices.empty())
          throw General_exception("no local storage devices");
        
        PLOG("Creating new session: device (size=%ld GB)",
             REDUCE_GB(_storage_devices[0]->nvme_device()->get_size_in_bytes(1)));
        
        _sessions.push_back(new Storage_agent_session(this,
                                                      arg0,
                                                      port,
                                                      _storage_devices[0],
                                                      _available_session_cores.back()));

        _available_session_cores.pop_back();
      }
      else {
        PWRN("someone whispered to me: %s %s %s:",arg0,arg1,arg2);
      }
              
      break;
    }
  case Message_group::EVENT_MEMBER_LEFT:
    {
      /* remove sessions for specific peer */
      _sessions.erase(std::remove_if(_sessions.begin(),
                                     _sessions.end(),
                                     [=](Storage_agent_session*& s) {
                                       if(s->peername().compare(arg0)==0) {
                                         _available_session_cores.push_back(s->core());
                                         delete s;                                         
                                         return true;
                                       }
                                       return false;
                                     }),
                      _sessions.end());
      break;
    }
  }

}

void Storage_agent::attach_devices()
{
  using namespace rapidjson;

  PLOG("Storage_agent attaching local devices..");
  
  std::string pciaddr;
  Value& addrs = config().device_pci_addr(member_id());
  for (SizeType i = 0; i < addrs.Size(); i++)
    _storage_devices.push_back(new Storage_device(addrs[i].GetString()));    
}

Nvme_device * Storage_agent::get_device(size_t index)
{
  assert(index < _storage_devices.size());
  return _storage_devices[index]->nvme_device();
}





