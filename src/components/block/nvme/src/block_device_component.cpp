/*
   Copyright [2017] [IBM Corporation]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include <set>
#include <core/dpdk.h>

#include "block_device_component.h"
#include "nvme_device.h"

using namespace Component;

static void __attribute__((constructor)) __Block_device_component()
{
  DPDK::eal_init(48);
}

static uint64_t issued_gwid __attribute__((aligned(8))) = 0;

Block_device_component::Block_device_component(const char * pci_addr, cpu_mask_t * cpus, Core::Poller * poller)
{
  if(cpus && poller)
    throw API_exception("conflicting parameters");
  
  if(cpus) {
    _device = new Nvme_device(pci_addr, *cpus);
  }
  else if(poller) {
    _device = new Nvme_device(pci_addr, poller);
  }
  else throw API_exception("missing parameters (cpus/poller)");
}



Block_device_component::~Block_device_component()
{
  delete _device;
}

workid_t
Block_device_component::
async_read(io_buffer_t buffer,
           uint64_t offset,
           uint64_t lba,
           uint64_t lba_count,
           int queue_id,
           io_callback_t cb,
           void * cb_arg0,
           void * cb_arg1)
{
  if(lba_count == 0)
    throw API_exception("bad async_read param (lba_count == 0)");
  
  void * ptr = (void*) (reinterpret_cast<char*>(buffer) + offset);
  assert(ptr);
  
  uint64_t gwid = ++issued_gwid;

  _device->queue_submit_async_op(ptr, lba, lba_count, COMANCHE_OP_READ, gwid,
                                 cb /* callback */, cb_arg0, cb_arg1, queue_id);

  return gwid;
}

workid_t
Block_device_component::
async_write(io_buffer_t buffer,
            uint64_t offset,
            uint64_t lba,
            uint64_t lba_count,
            int queue_id,
            io_callback_t cb,
            void * cb_arg0,
            void * cb_arg1)
{
  if(lba_count == 0)
    throw API_exception("bad async_write param");
  
  void * ptr = (void*) (reinterpret_cast<char*>(buffer)
                        + offset);
  assert(ptr);
  uint64_t gwid = ++issued_gwid;
  
  _device->queue_submit_async_op(ptr, lba, lba_count, COMANCHE_OP_WRITE, gwid,
                                 cb /* callback */, cb_arg0, cb_arg1, queue_id);

  return gwid;
}

bool
Block_device_component::
check_completion(uint64_t gwid, int queue_id)
{
  if(unlikely(gwid > issued_gwid))
    throw API_exception("%s: bad tag", __PRETTY_FUNCTION__);
  
  return _device->check_completion(gwid, queue_id);
}

void
Block_device_component::
attach_work(std::function<void(void*)> work_function, void * arg, int queue_id)
{
  _device->attach_work(queue_id, work_function, arg);
}

void
Block_device_component::
get_volume_info(VOLUME_INFO& devinfo)
{  
  strncpy(devinfo.volume_name,
          _device->get_device_id(),
          VOLUME_INFO_MAX_NAME);
  
  devinfo.block_size = _device->get_block_size(DEFAULT_NAMESPACE_ID);
  devinfo.max_lba = _device->get_size_in_blocks(DEFAULT_NAMESPACE_ID) - 1;
  devinfo.max_dma_len = _device->get_max_io_xfer_size();
}


/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Block_device_component_factory::component_id()) {
    printf("Creating 'Block_device_factory' component.\n");
    return static_cast<void*>(new Block_device_component_factory());
  }
  else return NULL;
}
