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


#pragma once
#ifndef __PART_GPT_TYPES_H__
#define __PART_GPT_TYPES_H__

#include <zlib.h>
#include <stdint.h>

#define _crc32(ptr,len) crc32(crc32(0L,Z_NULL,0L),(const Bytef*)(ptr),len)

const uint8_t partition_type_uuid[16] = {
  0xa2, 0xa0, 0xd0, 0xeb, 0xe5, 0xb9, 0x33, 0x44,
  0x87, 0xc0, 0x68, 0xb6, 0xb7, 0x26, 0x99, 0xc7,
};

const uint8_t partition_type_efi[16] = {
  0x28, 0x73, 0x2a, 0xc1, 0x1f, 0xf8, 0xd2, 0x11,
  0xba, 0x4b, 0x00, 0xa0, 0xc9, 0x3e, 0xc9, 0x3b,
};

const uint8_t partition_type_linux[16] = {
  0xa2, 0xa0, 0xd0, 0xeb, 0xe5, 0xb9, 0x33, 0x44,
  0x87, 0xc0, 0x68, 0xb6, 0xb7, 0x26, 0x99, 0xc7,
};

const uint8_t partition_type_swap[16] = {
  0x6d, 0xfd, 0x57, 0x06, 0xab, 0xa4, 0xc4, 0x43,
  0x84, 0xe5, 0x09, 0x33, 0xc8, 0x4b, 0x4f, 0x4f,
};

#define EFI_VERSION 0x00010000
#define EFI_MAGIC "EFI PART"
#define EFI_ENTRIES 128
#define EFI_NAMELEN 36

struct efi_header {
  uint8_t magic[8];
  uint32_t version;
  uint32_t header_sz;
  uint32_t crc32;
  uint32_t reserved;
  uint64_t header_lba;
  uint64_t backup_lba;
  uint64_t first_lba;
  uint64_t last_lba;
  uint8_t volume_uuid[16];
  uint64_t entries_lba;
  uint32_t entries_count;
  uint32_t entries_size;
  uint32_t entries_crc32;
} __attribute__((packed));


struct efi_entry {
  uint16_t type_uuid[8];
  uint8_t  uniq_uuid[16];
  uint64_t first_lba;
  uint64_t last_lba;
  uint64_t attr;
  uint16_t name[EFI_NAMELEN];
};

static_assert(sizeof(struct efi_entry)==128,"unexpected size of efi_entry");

#endif // __PART_GPT_TYPES_H__
