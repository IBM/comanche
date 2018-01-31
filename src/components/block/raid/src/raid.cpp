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

#include <common/exceptions.h>
#include <common/utils.h>
#include <api/raid_itf.h>
#include <rapidjson/document.h>

#include "raid.h"

using namespace Component;
using namespace rapidjson;

Raid_component::Raid_component()
{
}

Raid_component::~Raid_component()
{
  for(auto& i: _bdv_itf)
    i.block_device->release_ref();
}

//"{ "raidlevel" : 0 }"
void Raid_component::configure(std::string json_configuration)
{
  rapidjson::Document jdoc;
  jdoc.Parse(json_configuration.c_str());

  if(_bdv_itf.size() == 0)
    throw API_exception("must add devices before configuration");

  if(jdoc.FindMember("raidlevel") == jdoc.MemberEnd())
    throw API_exception("bad JSON configuration string for Raid_component");

  try {
    _raid_level = jdoc["raidlevel"].GetInt();
  }
  catch(...) {
    throw API_exception("bad JSON configuration string for Raid_component");
  }    
  PLOG("RAID - level set:%d across %d devices", _raid_level, _device_count);
  PLOG("RAID - max logical LBA is: %ld", _logical_block_count);
  PLOG("RAID - capacity %ld GB", REDUCE_GB(_logical_block_count * 4096));

  _ready = true;
}

void Raid_component::add_device(Component::IBlock_device * device, std::string role)
{
  if(_ready)
    throw API_exception("cannot add device after configuration");

  if(_device_count == MAX_DEVICE_COUNT)
    throw API_exception("too many devices");
  
  if(device==nullptr)
    throw API_exception("nullptr for add_device");

  VOLUME_INFO vi;
  device->get_volume_info(vi);
  if(vi.block_size != 4096)
    throw API_exception("raid device must be 4K block size");

  /* set lowest common max lba */
  if(_block_count == 0)
    _block_count = vi.block_count;
  else if(vi.block_count < _block_count)
    _block_count = vi.block_count;     
   
  device->add_ref();
  _bdv_itf.push_back({device,0}); /* ignore role for now */

  _device_count ++;
  _logical_block_count = _block_count * _device_count;
}

workid_t Raid_component::async_read(io_buffer_t buffer,
                                    uint64_t buffer_offset,
                                    uint64_t lba,
                                    uint64_t lba_count,
                                    int queue_id,
                                    io_callback_t cb,
                                    void * cb_arg0,
                                    void * cb_arg1)
{
  if(lba_count != 1)
    throw API_exception("invalid parameter(s)");

  unsigned index;
  IBlock_device * device = select_device(lba, index);
  assert(device);

  uint64_t gwid = device->async_read(buffer,
                                     buffer_offset,
                                     lba / _device_count,
                                     lba_count,
                                     queue_id,
                                     cb,
                                     cb_arg0,
                                     cb_arg1);
  uint64_t lgwid = (gwid | (((uint64_t)index) << 60));

  if(option_DEBUG)
    PLOG("issuing read to device[%u] lba=%ld count=%ld gwid=%lx lgwid=%lx",
         index, lba, lba_count, gwid, lgwid);
  
  return lgwid;
}

workid_t Raid_component::async_write(io_buffer_t buffer,
                                     uint64_t buffer_offset,
                                     uint64_t lba,
                                     uint64_t lba_count,
                                     int queue_id,
                                     io_callback_t cb,
                                     void * cb_arg0,
                                     void * cb_arg1)
{
  if(lba_count != 1)
    throw API_exception("invalid parameter(s)");

  unsigned index;
  IBlock_device * device = select_device(lba, index);
  assert(device);
  
  uint64_t gwid = device->async_write(buffer,
                                      buffer_offset,
                                      lba / _device_count,
                                      lba_count,
                                      queue_id,
                                      cb,
                                      cb_arg0,
                                      cb_arg1);

  uint64_t lgwid = (gwid | (((uint64_t)index) << 60));

  if(option_DEBUG)
    PLOG("issuing write to device[%u] lba=%ld count=%ld gwid=%lx lgwid=%lx",
         index, lba, lba_count, gwid, lgwid);

  return lgwid;
}

bool Raid_component::check_completion(workid_t lgwid, int queue_id)
{
  uint64_t index = lgwid >> 60;
  uint64_t gwid = lgwid & 0x00FFFFFFFFFFFFFFUL;
  if(option_DEBUG)
    PLOG("checking completion gwid=%lx lgwid=%lx", gwid, lgwid);
  
  return _bdv_itf[index].block_device->check_completion(gwid, queue_id);
}

uint64_t Raid_component::gwid_to_seq(uint64_t gwid)
{
  return (gwid & 0x00FFFFFFFFFFFFFFUL);
}

/** 
 * Get device information
 * 
 * @param devinfo pointer to VOLUME_INFO struct
 * 
 */
void Raid_component::get_volume_info(VOLUME_INFO& devinfo)
{
  devinfo.block_size = 4096;
  devinfo.block_count = _logical_block_count;
  sprintf(devinfo.volume_name,"SW-RAID-%d", _device_count);
}


/** 
 * Factory entry point
 * 
 */
extern "C" void * factory_createInstance(Component::uuid_t& component_id)
{
  if(component_id == Raid_component::component_id()) {
    PLOG("Creating 'Raid' component.");
    return static_cast<void*>(new Raid_component());
  }
  else return NULL;
}

