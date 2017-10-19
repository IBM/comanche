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

#ifndef __COMANCHE_CHANNEL_H__
#define __COMANCHE_CHANNEL_H__

#include <sstream>
#include <list>
#include <infiniband/verbs.h>
#include "buffer_manager.h"
#include "rdma_transport.h"
#include "bitset.h"
#include "types.h"


// forward decls
//
struct rte_ring;

/** 
 * Channel is the porting class. I don't want anything RDMA specific to leak out
 * through this API.
 */
class Channel
{
private:
  static constexpr size_t RECV_BUFFER_SIZE = KB(8);
  static constexpr size_t RECV_BUFFER_COUNT = 1024; /* must be power of 2 */
  static constexpr size_t HEADER_SIZE = 64; // cache line

public:
  
  enum {
    FLAGS_USE_DPDK=1,
  };

public:

  Channel(int core, int channel_flags = 0);
  ~Channel(); 

  status_t connect(const char * server_name,
                   const char * device_name = NULL,
                   int port = 18515);

  status_t wait_for_connect(const char * device_name = NULL,
                            int port = 18515);
  
  
  //  Channel::Channel_memory_handle alloc_handle();
  
  /** 
   * Release a handle from previous dpdk_recv call
   * 
   * @param memory Memory handle from 'dpdk_recv'
   */
  void release_handle(channel_memory_t memory);

  
  void abort() { _abort = true; }

  /** 
   * Get a pointer from a memory handle
   * 
   * @param handle 
   * 
   * @return Pointer to usable memory 
   */
  static inline void * ptr(channel_memory_t handle)
  {
    return static_cast<struct ibv_mr*>(handle)->addr;
  }


  inline struct ibv_mr * register_memory(void * contig_addr, size_t size)
  {
    return _transport.register_memory(contig_addr, size);
  }

  // inline size_t mr_size() const
  // {
  //   return _transport.mr_size();
  // }

  // inline void * header() const
  // {
  //   return _transport.header();
  // }

  inline status_t post_recv(uint64_t wid, channel_memory_t mr)
  {
    return _transport.post_recv(wid, mr);
  }

  
  inline status_t post_send(uint64_t wid,
                            channel_memory_t mr0,
                            channel_memory_t extra_mr = nullptr)
  {
    return _transport.post_send(wid, mr0, extra_mr);
  }

  // inline int post_send_single_mr(size_t sge_len,
  //                                int * complete_count,
  //                                Channel_memory_handle mr)
  // {
  //   return _transport.post_send_single_mr(sge_len, complete_count, mr);
  // }

  // inline status_t post_send_single_mr(size_t sge_len,
  //                                uint64_t wid,
  //                                Channel_memory_handle mr)
  // {
  //   return _transport.post_send_single_mr(sge_len, wid, mr);
  // }

  inline size_t poll_completions(std::function<void(uint64_t)> release_func)
  {
    return _transport.poll_completions(release_func);
  }

  // inline bool poll_completion_wid(uint64_t wid)
  // {
  //   return _transport.poll_completion_wid(wid);
  // }

  inline int outstanding()
  {
    return _transport.outstanding();
  }
    
  
  static std::string get_connection_info(int port)
  {
    std::stringstream ss;
    ss << "{ \"port\" : " << port << "}";
    return ss.str();
  }

  /** 
   * Allocate a memory region that can be used with DPDK/SPDK IO
   * 
   * @param size 
   * @param numa_socket 
   * 
   * @return 
   */
  struct ibv_mr * allocate_dpdk_mr(size_t size, int numa_socket = -1);

  struct ibv_mr * allocate_mr(size_t size, int numa_socket = -1);


private:

  void init_dpdk_memory(unsigned port);
  void init_memory(unsigned port);


  
private:

  struct rte_ring * _recv_mempool;
  RDMA_transport    _transport;
  bool              _use_dpdk;
  bool              _abort;
  int               _core;
};


#endif // __COMANCHE_CHANNEL_H__
