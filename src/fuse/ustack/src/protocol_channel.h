#ifndef __PROTOCOL_CHANNEL_H__
#define __PROTOCOL_CHANNEL_H__

#include <stdint.h>

enum {
  /* client request */
  IO_TYPE_READ = 1,
  IO_TYPE_WRITE = 2,

  /* server response*/
  IO_WRONG_TYPE = -1,
  IO_WRITE_FAIL = -3,
  IO_READ_FAIL = -4,
  IO_WRITE_OK = 3,
  IO_READ_OK = 4
};

struct IO_command
{
  uint8_t       type;
  uint8_t       flags;
  uint16_t      offset; // this corresponds to the actual physical addr
  uint16_t      sz_bytes; // requested  io size
  char          data[122];
}
__attribute__((packed));

#endif
