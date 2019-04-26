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


#include <common/logging.h>
#include <common/utils.h>
#include <common/dump_utils.h>
#include <common/exceptions.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "gpt.h"

namespace GPT {

Partition_table::
Partition_table(Component::IBlock_device * block_layer) :
  _bl(block_layer)
{
  TRACE();
  
  assert(_bl);
  _bl->add_ref();
  _bl->get_volume_info(_volinfo);
  
  _primary_gpt_iobuffer = _bl->allocate_io_buffer(_volinfo.block_size * 33,
                                                  _volinfo.block_size,
                                                  Component::NUMA_NODE_ANY);
  _secondary_gpt_iobuffer = _bl->allocate_io_buffer(_volinfo.block_size * 33,
                                                    _volinfo.block_size,
                                                    Component::NUMA_NODE_ANY);

  _bl->async_read(_primary_gpt_iobuffer, 0, 1, 33);
  uint64_t wid = _bl->async_read(_secondary_gpt_iobuffer, 0, 1, 33);
  while(!_bl->check_completion(wid));

  // {
  //   auto ptr = _bl->virt_addr(_primary_gpt_iobuffer);
  //   hexdump(ptr,KB(4));
  // }

  /* parse primary header and entries */
  _hdr = static_cast<struct efi_header *>(_bl->virt_addr(_primary_gpt_iobuffer));
  PINF("[GPT] number of partition entries: %u", _hdr->entries_count);
  PINF("[GPT] version: %u", _hdr->version);
  PINF("[GPT] single of entry: %u", _hdr->entries_size);
  PINF("[GPT] starting LBA: %lu", _hdr->entries_lba);

  if(strcmp((char*)_hdr->magic, EFI_MAGIC))
    throw Constructor_exception("invalid GPT table.");
  
  for(unsigned entry=0;entry<_hdr->entries_count;entry++) {
    struct efi_entry * e = get_entry(entry);
    if(e->type_uuid[0] == 0) continue;

    size_t size_in_mb = REDUCE_MB((e->last_lba - e->first_lba) * _volinfo.block_size);
    
    char tmpname[EFI_NAMELEN + 1] = {0};    
    for(unsigned i=0;i<EFI_NAMELEN;i++)
      tmpname[i] = e->name[i] & 0xFF;

    PINF("[GPT] - Partition:%u (%s) %ld-%ld (%ld MB)",
         entry, tmpname, e->first_lba, e->last_lba, size_in_mb);        
  }
}

Partition_table::
~Partition_table()
{
  _bl->free_io_buffer(_primary_gpt_iobuffer);
  _bl->free_io_buffer(_secondary_gpt_iobuffer);
  _bl->release_ref();
}


struct efi_entry *
Partition_table::
get_entry(unsigned index)
{
  if(index > _hdr->entries_count)
    throw General_exception("partition index exceeds limit");

  struct efi_entry * entry =
    reinterpret_cast<struct efi_entry *>(((char*) _hdr) + _volinfo.block_size);
  
  return &entry[index];
}


}
/* --- imported c functions --- */



namespace GPT
{

#if 0
void show(struct ptable *ptbl)
{
  struct efi_entry *entry = ptbl->entry;
  unsigned n, m;
  char name[EFI_NAMELEN + 1];

  fprintf(stderr,"ptn  start block   end block     name\n");
  fprintf(stderr,"---- ------------- ------------- --------------------\n");

  for (n = 0; n < EFI_ENTRIES; n++, entry++) {
    if (entry->type_uuid[0] == 0)
      break;
    for (m = 0; m < EFI_NAMELEN; m++) {
      name[m] = entry->name[m] & 127;
    }
    name[m] = 0;
    fprintf(stderr,"#%03d %13ld %13ld %s\n",
            n + 1, entry->first_lba, entry->last_lba, name);
  }
}

u64 find_next_lba(struct ptable *ptbl)
{
  struct efi_entry *entry = ptbl->entry;
  unsigned n;
  u64 a = 0;
  for (n = 0; n < EFI_ENTRIES; n++, entry++) {
    if ((entry->last_lba + 1) > a)
      a = entry->last_lba + 1;
  }
  return a;
}

u64 next_lba = 0;

u64 parse_size(char *sz)
{
  int l = strlen(sz);
  u64 n = strtoull(sz, 0, 10);
  if (l) {
    switch(sz[l-1]){
    case 'k':
    case 'K':
      n *= 1024;
      break;
    case 'm':
    case 'M':
      n *= (1024 * 1024);
      break;
    case 'g':
    case 'G':
      n *= (1024 * 1024 * 1024);
      break;
    }
  }
  return n;
}


int parse_ptn(struct ptable *ptbl, char *x)
{
  const u8 *type = partition_type_uuid; 
  char *y = strchr(x, ':');
  char *z;
  u64 sz;

  if (!y) {
    fprintf(stderr,"invalid partition entry: %s\n", x);
    return -1;
  }
  *y++ = 0;

  z = strchr(y, '=');
  if (z) {
    *z++ = 0;
    if (!strcmp(z, "efi")) {
      type = partition_type_efi;
    } else if (!strcmp(z, "linux")) {
      type = partition_type_linux;
    } else if (!strcmp(z, "swap")) {
      type = partition_type_swap;
    }
  }
  if (*y == 0) {
    sz = ptbl->header.last_lba - next_lba;
  } else {
    sz = parse_size(y);
    if (sz & 511) {
      fprintf(stderr,"partition size must be multiple of 512\n");
      return -1;
    }
    sz /= 512;
  }

  if (sz == 0) {
    fprintf(stderr,"zero size partitions not allowed\n");
    return -1;
  }

  if (x[0] && add_ptn(ptbl, next_lba, next_lba + sz - 1, x, type))
    return -1;

  next_lba = next_lba + sz;
  return 0;
}


void update_crc32(struct ptable *ptbl) {
  u32 n;


  n = _crc32((void*) ptbl->entry, sizeof(ptbl->entry));
  ptbl->header.entries_crc32 = n;

  ptbl->header.crc32 = 0;
  n = _crc32((void*) &ptbl->header, sizeof(ptbl->header));
  ptbl->header.crc32 = n;
}

#endif

} // GPT namespace


