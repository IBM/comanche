#ifndef __PROTOCOL_CHANNEL_H__
#define __PROTOCOL_CHANNEL_H__

#include <stdint.h>

enum {
  IO_TYPE_READ = 1,
  IO_TYPE_WRITE = 1,
};

struct IO_command
{
  uint8_t       type;
  uint8_t       flags;
  uint16_t      offset;
  char          data[124];
}
__attribute__((packed));

#endif
