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


/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <api/block_itf.h>
#include <rapidjson/document.h>

#include "block_unvme.h"

enum {
  IOCTL_CMD_GETBITMAP = 9,
  IOCTL_CMD_GETPHYS = 10,
};

typedef struct
{
  addr_t vaddr;
  addr_t out_paddr;
} __attribute__((packed)) IOCTL_GETPHYS_param;


using namespace Component;

Block_unvme::Block_unvme(std::string config) : _size_in_blocks(0)
{
  _ns = unvme_openq(config.c_str(), QUEUE_COUNT, QUEUE_SIZE);
  if(!_ns) throw Constructor_exception("unable to open NVMe device (%s)",
                                       config.c_str());

  VOLUME_INFO vi;
  get_volume_info(vi);

  PINF("Volume: block_size=%u", vi.block_size);
  PINF("        device_size=%ld GB", REDUCE_GB(vi.max_lba * vi.block_size));
  PINF("        maxlba=%ld", vi.max_lba);
  PINF("        name=(%s)", vi.volume_name);
  PINF("        maxqueuelen=%u", _ns->maxiopq);

}
 
Block_unvme::~Block_unvme()
{
  unvme_close(_ns);
}


/** 
 * Factory 
 * 
 */
Component::IBlock_device *
Block_unvme_factory::
create(std::string config_string, cpu_mask_t * cpus, Core::Poller * poller)
{
  if(cpus || poller)
    throw API_exception("unsupported");
  
  IBlock_device *blk = static_cast<IBlock_device*>(new Block_unvme(config_string));
  blk->add_ref();
  return blk;
}


/** 
 * IBlock_device
 * 
 */

// void
// Block_unvme::
// read(Component::io_buffer_t buffer,
//      uint64_t buffer_offset,
//      uint64_t lba,
//      uint64_t lba_count)
// {
//   byte * bptr = reinterpret_cast<byte*>(buffer);
//   bptr += buffer_offset;
  
//   int rc = unvme_read(_ns,
//                       0 /* qid */,
//                       bptr,
//                       lba,
//                       lba_count);
//   if(rc)
//     throw API_exception("unvme_read failed");
// }

// void
// Block_unvme::
// write(Component::io_buffer_t buffer,
//       uint64_t buffer_offset,
//       uint64_t lba,
//       uint64_t lba_count)
// {
//   byte * bptr = reinterpret_cast<byte*>(buffer);
//   bptr += buffer_offset;
  
//   int rc = unvme_write(_ns,
//                       0 /* qid */,
//                       bptr,
//                       lba,
//                       lba_count);
//   if(rc)
//     throw API_exception("unvme_write failed");
// }


workid_t
Block_unvme::
async_read(io_buffer_t buffer,
           uint64_t buffer_offset,
           uint64_t lba,
           uint64_t lba_count,
           int queue_id,
           io_callback_t cb,
           void * cb_arg0,
           void * cb_arg1)
{
  if(option_DEBUG)
    PINF("block-unvme: async_read(buffer=%p, offset=%lu, lba=%lu, lba_count=%lu",
         (void*) buffer, buffer_offset, lba, lba_count);

  if(cb || cb_arg0 || cb_arg1)
    throw API_exception("UNVMe device does not yet support callbacks");

  if(queue_id > 0)
    throw API_exception("invalid queue identifier");

  byte * bptr = reinterpret_cast<byte*>(buffer);
  bptr += buffer_offset;

  unvme_iod_t iod = unvme_aread(_ns,
                                0 /* qid */,
                                bptr,
                                lba,
                                lba_count);

  if(iod == nullptr)
    throw General_exception("unvme_aread failed");

  return reinterpret_cast<workid_t>(iod);
}
 
workid_t
Block_unvme::
async_write(io_buffer_t buffer,
            uint64_t buffer_offset,
            uint64_t lba,
            uint64_t lba_count,
            int queue_id,
            io_callback_t cb,
            void * cb_arg0,
            void * cb_arg1)
{
  if(option_DEBUG)
    PINF("block-unvme: async_write(buffer=%p, offset=%lu, lba=%lu, lba_count=%lu",
         (void*)buffer, buffer_offset, lba, lba_count);

  if(cb || cb_arg0 || cb_arg1)
    throw API_exception("UNVMe device does not yet support callbacks");

  if(queue_id > 0)
    throw API_exception("invalid queue identifier");

  byte * bptr = reinterpret_cast<byte*>(buffer);
  bptr += buffer_offset;

  unvme_iod_t iod = unvme_awrite(_ns,
                                 0 /* qid */,
                                 bptr,
                                 lba,
                                 lba_count);
  if(iod == nullptr)
    throw General_exception("unvme_awrite failed");
  
  return reinterpret_cast<workid_t>(iod);
}

/** 
 * Check for completion of a work request. This API is thread-safe.
 * 
 * @param gwid Work request identifier
 * @param queue_id Logical queue identifier (not used)
 * 
 * @return True if completed.
 */
bool
Block_unvme::
check_completion(workid_t gwid, int queue_id)
{
  if(gwid == 0)
    throw API_exception("check all gwid=0 not supported");

  if(queue_id > 0)
    throw API_exception("invalid queue identifier");

  unvme_iod_t iod = reinterpret_cast<unvme_iod_t>(gwid);
  assert(iod);
  int rc = unvme_apoll(iod, 0/* timeout */);
  if(rc==0) return true;
  return false;
}

/** 
 * Get device information
 * 
 * @param devinfo pointer to VOLUME_INFO struct
 * 
 * @return S_OK on success
 */
void
Block_unvme::
get_volume_info(VOLUME_INFO& devinfo)
{
  devinfo.block_size = _ns->blocksize;
  devinfo.distributed = false;
  devinfo.hash_id = 0;
  devinfo.max_lba = _ns->blockcount;
  strncpy(devinfo.volume_name,_ns->device,16);
}



/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Block_unvme_factory::component_id()) {
    PLOG("Creating 'Block_unvme' factory.");
    return static_cast<void*>(new Block_unvme_factory());
  }
  else
    return NULL;
}

