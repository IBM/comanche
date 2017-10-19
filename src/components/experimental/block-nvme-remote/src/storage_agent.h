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

#ifndef __COMANCHE_STORAGE_AGENT_H__
#define __COMANCHE_STORAGE_AGENT_H__

#include <atomic>
#include <list>
#include <vector>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <common/logging.h>
#include <stdint.h>

#include "json_config.h"
#include "messaging.h"
#include "agent.h"
#include "channel.h"
#include "storage_device.h"


class Nvme_device;
class Nvme_queue;
class Storage_agent_session;

/** 
 * Storage agent sits on a storage node.  Multiple storage agents
 * serve a volume agent.
 * 
 */

class Storage_agent : public Agent
{
private:
  static constexpr bool option_DEBUG = true;

public:
  /** 
   * Constructor
   * 
   * @param configuration_file Configuration file
   * 
   * @return 
   */
  Storage_agent(const char * configuration_file);
  virtual ~Storage_agent();

  void messaging_callback(int type,
                          const char *arg0,
                          const char *arg1,
                          const char *arg2);
  

private:


  void attach_devices();
  Nvme_device * get_device(size_t index);

  std::vector<Storage_agent_session *> _sessions; /**< map of sessions; key is client host name */
  std::vector<Storage_device*>         _storage_devices;

  std::vector<int> _available_session_cores;
  // temporary
  //  Local_volume * _lv;

  
};


#endif // __COMANCHE_STORAGE_AGENT_H__
