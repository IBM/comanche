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

#include <memory>
#include <common/utils.h>
#include <rte_lcore.h>

#include "policy_mirror.h"
#include "policy_local.h"

#include "volume_agent.h"
#include "block_service.h"
#include "storage_device.h"
#include "eal_init.h"
#include "logical_volume.h"


/** 
 * Implementation
 * 
 *  
 */

Block_service::Block_service(const char * configuration_file) :
  _cfg(configuration_file)
{
  assert(configuration_file);
  
  PLOG("client: config=%s", configuration_file);

  DPDK::eal_init(32); /* 32 MB memory limit */

  _local_devices = std::unique_ptr<Local_devices>
    (new Local_devices(configuration_file));

  switch(_cfg.policy()) {
  case Json_configuration::POLICY_MIRROR:
    _volume_agents.push_back(new Volume_agent(configuration_file));
    break;
  case Json_configuration::POLICY_LOCAL:
    break;
  }
}

Block_service::~Block_service()
{
  for(auto& i: _volume_agents) delete i;
}

Logical_volume* Block_service::create_volume(const char * volume_name)
{
  assert(volume_name);
  
  /* TODO: check existing volumes */
  Logical_volume * newvol = new Logical_volume(volume_name);
  _logical_volumes.push_back(newvol);

  PINF("new volume (%s) created.", volume_name);
  return newvol;
}

void Block_service::attach_local_device(Logical_volume* lvol, int index)
{
  auto dev = _local_devices->device(index);
  if(!dev)
    throw API_exception("invalid device index (%d) in attach_local", index);

  lvol->add_policy(new Policy_local(dev));
}

void Block_service::attach_lr_mirror(Logical_volume* lvol, int index, unsigned core)
{
  auto dev = _local_devices->device(index);
  if(!dev)
    throw API_exception("invalid device index (%d) in attach_lr_mirror", index);

  /* for the moment volume agent was created from the initial
     configuration file */
  lvol->add_policy(new Policy_mirror(dev,
                                    /* va session will be delete by volume agent */
                                    _volume_agents[0]->create_session(core)));
}

void Block_service::delete_volume(Logical_volume* volume)
{
  for(std::vector<Logical_volume*>::iterator i = _logical_volumes.begin();
      i != _logical_volumes.end(); i++) {    
    if(volume == *i) {
      delete *i;
      _logical_volumes.erase(i);
      return;
    }
  }
  throw API_exception("invalid volume handle: delete_volume");
}


Block_service_session* Block_service::create_session(Logical_volume* lvol,
                                                     unsigned core)
{
  Block_service_session * session = new Block_service_session(lvol->get_policy(),
                                                              core);
  _sessions.push_back(session);
  PLOG("new Block_service_session: %p  vol=%s", session, lvol->name());
  return session;
}

void Block_service::close_session(Block_service_session* session)
{
  assert(session);
  delete session;
}


