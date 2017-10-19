#ifndef __COMPONENT_TYPES_H__
#define __COMPONENT_TYPES_H__

#include <stdint.h>

/* general types */
typedef uint64_t lba_t;
typedef int64_t  index_t;

enum {
  FLAGS_CREATE = 0x1,
  FLAGS_FORMAT = 0x2, /*< force region manager to format block device */
};

#endif
