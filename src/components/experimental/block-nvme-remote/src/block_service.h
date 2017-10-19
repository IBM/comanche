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

#ifndef __COMANCHE_CLIENT_H__
#define __COMANCHE_CLIENT_H__

#include <memory>

#include "types.h"
#include "block_service_session.h"
#include "logical_volume.h"

class Volume_agent;

/** 
 * Private class to manage local devices
 * 
 */
class Local_devices
{
public:
  Local_devices(const char * configuration_file) : _cfg(configuration_file) {

    using namespace rapidjson;

    char hostname[HOST_NAME_MAX];
    check_zero(gethostname(hostname, HOST_NAME_MAX));
    
    /* create NVMe device instances according to configuration */
    Value& addrs = _cfg.device_pci_addr(hostname);
    for (SizeType i = 0; i < addrs.Size(); i++) {
      auto pci_addr = addrs[i].GetString();
      _devices[pci_addr]  = new Storage_device(pci_addr);
    }    
  }

  ~Local_devices() {
    TRACE();
    for(auto& device: _devices) {
      delete device.second;
    }
  }

  Storage_device * device(unsigned idx) const
  {
    unsigned i=0;
    for(auto& device: _devices) {
      if(idx == i) return device.second;
      i++;
    }
    return nullptr;
  }

private:
  
  Json_configuration           _cfg;
  std::map<std::string, Storage_device*> _devices;
};


/** 
 * User/developer facing class.  Includes are minimal.
 * 
 */
class Block_service
{
public:
  
  Block_service(const char * configuration_file);
  
  virtual ~Block_service();

  Logical_volume* create_volume(const char * volume_name);

  void delete_volume(Logical_volume* lvol);

  void attach_local_device(Logical_volume* lvol, int index);

  void attach_lr_mirror(Logical_volume* lvol, int index, unsigned core); // temp

  /** 
   * Creating a session basically sets up a thread to service
   * requests for the given logic volume.  The worker thread is
   * assigned to the specified core.
   * 
   * @param lvol Logical volume to set up session for
   * @param core Core to assign worker thread
   * 
   * @return Reference to a block service session (clean up with close_session)
   */
  Block_service_session* create_session(Logical_volume* lvol, unsigned core);

  void close_session(Block_service_session* session);

  Logical_volume * logical_volume(unsigned index) {
    return _logical_volumes[index];
  }
  
private:

  static constexpr unsigned MAX_LCORE = 46;
  static constexpr unsigned SERVICE_QUEUE_DEPTH = 128;

  std::unique_ptr<Local_devices> _local_devices; // raw local nvme devices
  std::vector<Volume_agent *>    _volume_agents;

  Json_configuration _cfg;

  std::vector<Logical_volume *>        _logical_volumes;
  std::vector<Block_service_session *> _sessions;
};




#endif
