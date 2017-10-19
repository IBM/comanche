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

#ifndef __COMANCHE_VOLUME_AGENT__
#define __COMANCHE_VOLUME_AGENT__

#include <memory>
#include <stack>
#include <set>
#include <thread>
#include <mutex>
#include <list>
#include <stdint.h>

#include "volume_agent_session.h"
#include "json_config.h"
#include "messaging.h"
#include "agent.h"
#include "channel.h"
#include "block_service.h"
#include "bitset.h"

class Volume_agent_session;

/** 
 * Volume agent sits on the client side.
 * 
 */
class Volume_agent : public Agent
{
private:
  static constexpr bool option_DEBUG = false;

public:
  /** 
   * Constructor
   * 
   * @param configuration_file JSON configuration file
   * 
   */
  Volume_agent(const char * configuration_file);

  /** 
   * Destrucor
   * 
   */
  virtual ~Volume_agent() noexcept;

  Volume_agent_session * create_session(unsigned core);
  
private:
  void messaging_callback(int type,
			  const char *arg0,
			  const char *arg1,
			  const char *arg2);

  static void __va_callback_trampoline(void *cbarg,
                                       int type,
                                       const char *arg0,
                                       const char *arg1,
                                       const char *arg2);

  std::set<unsigned> _session_cores;
  std::vector<Volume_agent_session*> _sessions;
};


#endif
