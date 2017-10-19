
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

#ifndef __COMANCHE_PROTOCOLS_H__
#define __COMANCHE_PROTOCOLS_H__

#include <stdint.h>
#include "types.h"


static constexpr uint32_t COMANCHE_PROTOCOL_MAGIC = 0xAB1A2E1; // Ablaze!

enum {
  COMANCHE_PROTOCOL_FLAG_REQUEST = 0x1,
  COMANCHE_PROTOCOL_FLAG_RESPONSE = 0x2,
  COMANCHE_PROTOCOL_OP_READ=0x2, // do not modify (see Nvme_queue.h)
  COMANCHE_PROTOCOL_OP_WRITE=0x4,
  COMANCHE_PROTOCOL_FLAG_SIGNAL_CALLER = 0x8,
  COMANCHE_PROTOCOL_FLAG_CHECK_COMPLETION = 0x10,
  COMANCHE_PROTOCOL_FLAG_MULTISEG = 0x20, /* multiple segments in this command */
  //  COMANCHE_PROTOCOL_OP_QUERY=0x40;
};

struct IO_command;

struct IO_command
{ 
  uint32_t magic;
  uint32_t op_flags; /* OP and flags; could use a bitfield */
  uint64_t len;
  uint64_t offset;
  uint64_t lba;  
  uint64_t lba_count;
  union {
    uint64_t gwid; /* global work id */
    uint64_t mrdesc; /* memory buffer descriptor */
  };
  uint64_t md_mrdesc;

} __attribute__((packed));


#endif
