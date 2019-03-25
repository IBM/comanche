/*
   Copyright [2017-2019] [IBM Corporation]
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

#ifndef __API_RAID_H__
#define __API_RAID_H__

#include <api/raid_itf.h>
#include <api/block_itf.h>

namespace Component
{

class IRaid : public IBlock_device
{
public:
  DECLARE_INTERFACE_UUID(0xea738dc7,0x063e,0x45df,0xa111,0xb4,0x15,0x73,0x8f,0x2d,0x9b);


  /** 
   * Configure RAID
   * 
   * @param json_configuration Configuration string
   */
  virtual void configure(std::string json_configuration) = 0;


  /** 
   * Add block device to RAID array
   * 
   * @param device Block device
   * @param role JSON-defined role
   */
  virtual void add_device(Component::IBlock_device * device, std::string role = "") = 0;

  /** 
   * Convert logical gwid (embedding device id) to sequential gwid
   * 
   * @param lgwid Logical gwid
   * 
   * @return Sequential gwid
   */
  virtual uint64_t gwid_to_seq(uint64_t gwid) = 0;
};

} // Component

#endif // __API_RAID_H__
