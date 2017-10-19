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

#ifndef __COMANCHE_TYPES_H__
#define __COMANCHE_TYPES_H__

#include <common/exceptions.h>
#include <rapidjson/stream.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/istreamwrapper.h>
#include <infiniband/verbs.h>
#include <rte_memory.h>
#include <rte_mempool.h>
#include <rte_ring.h>
#include <rte_errno.h>

#include "config.h"
#include "protocols.h"

typedef rapidjson::Document json_document_t;
typedef struct ibv_mr *     channel_memory_t;
typedef uint64_t            io_memory_t;

typedef void (*io_callback_t)(int,void*);

enum {
  COMANCHE_OP_READ=0x2, // do not modify (see Nvme_queue.h)
  COMANCHE_OP_WRITE=0x4,
};

enum { // operation modes
  MODE_UNKNOWN=0,
  MODE_DIRECT=1, /**< directly access HW queues */
  MODE_QUEUED=2, /**< software lockfree queues in front of HW queue */
};

 
class Nvme_queue;

typedef struct {
  void * buffer;
  uint64_t lba;
  uint64_t lba_count;
  io_callback_t cb;
  void * arg;
  int op;
  int tag;
  uint32_t magic;
  Nvme_queue * queue;
#ifdef CONFIG_QUEUE_STATS
  cpu_time_t time_stamp;
#endif
} queued_io_descriptor_t;


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
