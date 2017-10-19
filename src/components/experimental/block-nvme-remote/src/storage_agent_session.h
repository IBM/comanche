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

#ifndef __STORAGE_AGENT_SESSION_H__
#define __STORAGE_AGENT_SESSION_H__

#include <nvme_queue.h>
#include <thread>
#include "storage_device.h"
#include "storage_agent.h"


/** 
 * Client session class
 * 
 */
class Storage_agent_session
{
  static constexpr bool option_DEBUG = false;
  
public:
  Storage_agent_session(Storage_agent * parent,
                        const char * peer_name,
                        int port,
                        Storage_device* storage_device,
                        unsigned core);

  virtual ~Storage_agent_session();

  const char * hostname() {
    static char hostname[HOST_NAME_MAX];
    check_zero(gethostname(hostname, HOST_NAME_MAX));
    return hostname;
  }

  std::string peername() {
    return _peer_name;
  }

  inline unsigned core() const {
    return _core;
  }

private:
  void session_thread(int port, unsigned core);

  std::string      _peer_name;
  Storage_agent *  _sa;
  Storage_device * _storage_device;
  Nvme_queue *     _queue;
  std::thread *    _thread;
  bool             _exit;
  unsigned         _core;
}; 



#endif // __STORAGE_AGENT_SESSION_H__
