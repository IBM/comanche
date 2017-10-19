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

#include <nvme_device.h>
#include <nvme_queue.h>

#include "storage_device.h"


Storage_device::Storage_device(const char* pci_addr)
{
  PLOG("creating new Storage_device (%s)", pci_addr);
  _nvme_device = new Nvme_device(pci_addr, MODE_DIRECT);
  if(!_nvme_device)
    throw General_exception("unable to instantiate Nvme_device");
  
  assert(_nvme_device);

  for(unsigned i=0; i<NUM_QUEUES_TO_ALLOCATE; i++) {
    Nvme_queue * qp = _nvme_device->allocate_io_queue_pair();
    assert(qp);
    _available_nvme_queues.push_back(qp);
  }
  
}

Storage_device::~Storage_device() {
  for(auto& qp : _available_nvme_queues)
    delete qp;

  delete _nvme_device;
}


