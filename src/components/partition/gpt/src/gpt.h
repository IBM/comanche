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
