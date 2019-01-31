#ifndef __NUPM_PMEM_LOW_LEVEL_H__
#define __NUPM_PMEM_LOW_LEVEL_H__

#include "x86_64/flush.h"

namespace nupm
{
  inline static void mem_flush(const void *addr, size_t len) {
    flush_clflushopt_nolog(addr,len);
  }
    
}

#endif
