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

#ifndef __COMANCHE_STORAGE_DEVICE_H__
#define __COMANCHE_STORAGE_DEVICE_H__

#include <list>


class Nvme_device;
class Nvme_queue;


class Storage_device
{
private:
  static constexpr unsigned NUM_QUEUES_TO_ALLOCATE = 4; // up to 32
  
public:
  Storage_device(const char * pci_addr);    
  virtual ~Storage_device();

  /** 
   * Get interface to NVMe storage device.
   * 
   * 
   * @return 
   */
  inline Nvme_device * nvme_device() { return _nvme_device; }

  /** 
   * Allocate a hardware queue on a storage device.
   * 
   * 
   * @return Pointer to queue (client should free with free_queue).
   */
  Nvme_queue * allocate_queue() {
    if(_available_nvme_queues.empty()) return nullptr;
    Nvme_queue * q = _available_nvme_queues.back();
    _available_nvme_queues.pop_back();
    return q;
  }

  /** 
   * Release a hardware queue
   * 
   * @param queue Pointer previously allocated with allocate_queue
   */
  void free_queue(Nvme_queue * queue) {
    _available_nvme_queues.push_back(queue);
  }
  
private:
  
  Nvme_device *          _nvme_device;
  std::list<Nvme_queue*> _available_nvme_queues;
};



#endif // __COMANCHE_STORAGE_DEVICE_H__
