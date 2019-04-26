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

#ifndef __PART_GPT_GPT_H__
#define __PART_GPT_GPT_H__

#include <api/block_itf.h>
#include "gpt_types.h"

namespace GPT
{

class Partition_table
{
public:

  Partition_table(Component::IBlock_device * block_layer);

  virtual ~Partition_table();

  struct efi_entry * get_entry(unsigned index);

  struct efi_header * hdr() const { return _hdr; }
  
private:
  Component::IBlock_device * _bl;
  Component::VOLUME_INFO _volinfo;
  Component::io_buffer_t _primary_gpt_iobuffer;
  Component::io_buffer_t _secondary_gpt_iobuffer;

  struct efi_header * _hdr;
};

}

#endif // __PART_GPT_GPT_H__
