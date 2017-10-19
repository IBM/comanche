/*
   Copyright [2017] [IBM Corporation]

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

#ifndef __COMANCHE_TYPES_H__
#define __COMANCHE_TYPES_H__

#include <common/exceptions.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <infiniband/verbs.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_ring.h>
#include <rte_errno.h>

#include "config.h"

typedef rapidjson::Document json_document_t;
typedef struct ibv_mr *     channel_memory_t;
typedef uint64_t            io_memory_t;

typedef void (*io_callback_t)(uint64_t,void*,void*);

enum {
  COMANCHE_OP_READ = 0x2, // do not modify (see Nvme_queue.h)
  COMANCHE_OP_WRITE = 0x4,
  COMANCHE_OP_CHECK_COMPLETION = 0x8,
};

enum {
  IO_STATUS_UNKNOWN  = 0,
  IO_STATUS_COMPLETE = 1,
  IO_STATUS_PENDING  = 2,
  IO_STATUS_FAILED   = 3,
};

 
class Nvme_queue;

class IO_descriptor;

class IO_descriptor
{
public:

  IO_descriptor() : prev(nullptr), next(nullptr)
  {
  }
  
  IO_descriptor * prev;
  IO_descriptor * next;
  void * buffer;
  union {
    uint64_t lba;
    volatile uint64_t status;
  };
  uint64_t lba_count;
  io_callback_t cb;
  void * arg0;
  void * arg1;

  int op;
  uint64_t tag;
  Nvme_queue * queue;
#ifdef CONFIG_QUEUE_STATS
  cpu_time_t time_stamp;
#endif
};


namespace DPDK
{

template <typename T>
class Memory_pool
{
public:
  Memory_pool(const char * name, size_t n_elements, int flags=0) {

    if(flags == 0)
      flags = MEMPOOL_F_SP_PUT | MEMPOOL_F_SC_GET;
    
    _mem_pool = rte_mempool_create(name,
                                   n_elements,
                                   sizeof(T),
                                   0,
                                   0,
                                   NULL, NULL, NULL, NULL,
                                   SOCKET_ID_ANY,
                                   flags);
    if(!_mem_pool)
      throw Constructor_exception("Memory_pool: rte_mempool_create failed");
  }

  ~Memory_pool() {
    rte_mempool_free(_mem_pool);
  }

  inline T* alloc() {
    T* p = nullptr;
    int rc = rte_mempool_get(_mem_pool,(void**)&p);
    if(rc) {
      PWRN("Memory_pool::alloc underflow");
      return nullptr;
    }
    return p;
  }

  inline void free(T* element) {
    rte_mempool_put(_mem_pool, element);
  }
  
private:
  struct rte_mempool * _mem_pool;
};


}

#endif
