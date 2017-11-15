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

#ifndef __COMANCHE_RDMA_TRANSPORT_H__
#define __COMANCHE_RDMA_TRANSPORT_H__

#include <thread>
#include <mutex>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <common/logging.h>
#include <common/exceptions.h>
#include <common/utils.h>
#include <assert.h>
#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>

#include "rdma_helper.h"

/** 
 * Rdma transport based on RC connection type over RoCE v2
 * 
 */
class Rdma_transport
{
private:
  static const size_t PRIMARY_SGE_MEM_SIZE = 4096*2;
  static const int QP_RX_DEPTH = 256;
  static const int QP_TX_DEPTH = 256;
  static const int TX_WATERMARK = QP_TX_DEPTH - 16;
  static const int IB_PORT = 1;
  static const int GIDX = 3; // see show_gids tool
  static const enum ibv_mtu MTU = IBV_MTU_1024;
  int _sl = 0;

  static const bool option_DEBUG = true;
  
public:

  /** 
   * Constructor
   * 
   * @param port TCP port for connection establishment
   * @param server_name IP hostname of server (NULL assumes initiator/client role)
   * @param device_name Name of Rdma NIC device (NULL uses first found device)
   */
  Rdma_transport() :
    _device(NULL),
    _ctx(NULL),
    _outstanding(0),
    _complete_ctr(0)
  {
  }

  /** 
   * Wait for connection
   * 
   * @param port 
   * 
   * @return 
   */
  status_t wait_for_connect(const char * device_name = NULL, int port = 18515)
  {
    return connect(NULL, device_name, port);
  }
  
  /** 
   * Connect to a peer
   * 
   * @param port 
   * @param server_name 
   * @param device_name 
   */
  status_t connect(const char * device_name,
                   const char * server_name,
                   int port = 18515)
  {
    /* open device */    
    _device = open_device(device_name);
    assert(_device);
    
    /* init context */
    _ctx = channel_init_ctx(_device,
                            IBV_QPT_RC, /* reliable (with flow-control), connection oriented */
                            PRIMARY_SGE_MEM_SIZE,
                            QP_RX_DEPTH,
                            QP_TX_DEPTH,
                            IB_PORT, /* port */
                            false, /* use_event */
                            1/*inlr_recv */);
    assert(_ctx);

    /* get port info */
    if (rdma_get_port_info(_ctx->context, IB_PORT, &_ctx->portinfo)) {
      PERR("Couldn't get port info");
      return E_FAIL;
    }


    /* configure local endpoint */
    _local_endpoint.lid = _ctx->portinfo.lid;
    if (_ctx->portinfo.link_layer != IBV_LINK_LAYER_ETHERNET &&
        !_local_endpoint.lid) {
      PERR("Couldn't get local LID\n");
      return E_FAIL;
    }

    if (ibv_query_gid(_ctx->context, IB_PORT, GIDX, &_local_endpoint.gid)) {
			PERR("Can't read sgid of index %d\n", GIDX);
      return E_FAIL;
    }

    _local_endpoint.qpn = _ctx->qp->qp_num;
    _local_endpoint.psn = lrand48() & 0xffffff;
    inet_ntop(AF_INET6, &_local_endpoint.gid, _gid, sizeof _gid);
    PLOG("  local address:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s",
	       _local_endpoint.lid, _local_endpoint.qpn, _local_endpoint.psn, _gid);

    /* exchange connection information over TCP/IP */
    
    int retries = 6;
    _remote_endpoint = NULL;

    while(retries > 0 && _remote_endpoint == NULL) {

      if(!server_name) {
        _remote_endpoint = channel_server_exch_dest(_ctx,
                                                    IB_PORT, MTU, port,
                                                    _sl,
                                                    &_local_endpoint,
                                                    GIDX);
      }
      else {
        _remote_endpoint = channel_client_exch_dest(server_name, port, &_local_endpoint);
      }
        
      if(_remote_endpoint == NULL) {
        sleep(1);
        retries--;
        PLOG("retrying channel info exchange");
      }          
    }

    if(!_remote_endpoint) {
      PERR("failed to connect to peer :-(");
      return E_FAIL;
    }
    
    inet_ntop(AF_INET6, &_remote_endpoint->gid, _gid, sizeof _gid);
    PLOG("  remote address: LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n",
	       _remote_endpoint->lid, _remote_endpoint->qpn, _remote_endpoint->psn, _gid);

    if (server_name) {
      if (channel_connect_ctx(_ctx,
                              IB_PORT,
                              _local_endpoint.psn,
                              MTU,
                              _sl,
                              _remote_endpoint,
                              GIDX)) {
        PERR("failed to connect Rdma channel");
        return E_FAIL;
      }
    }

    return S_OK;
  }

  /** 
   * Destructor
   * 
   * 
   * @return 
   */
  virtual ~Rdma_transport()
  {
    for(auto& mr : _mr_vector) {
      ibv_dereg_mr(mr);
    }
    channel_close_ctx(_ctx);
  }

  /** 
   * Get a pointer to the first segment buffer area
   * 
   * 
   * @return 
   */
  inline void * header() const
  {
    return (void*) nullptr; //_ctx->buf;
  }

  /** 
   * Get size of implicit memory region
   * 
   * 
   * @return 
   */
  // inline size_t mr_size() const
  // {
  //   return _ctx->mr->length;
  // }

  /** 
   * Register memory with the context.  This is typically
   * done to register the DPDK buffers for Rdma transmission.
   * 
   * @param contig_addr Pointer to memory with contiguous physical
   * @param len Length of memory in bytes
   * 
   * @return Pointer to IBV memory region
   */
  struct ibv_mr * register_memory(void * contig_addr, size_t size)
  {
    struct ibv_mr * mr = channel_register_memory(_ctx, contig_addr, size);
    assert(mr);
    _mr_vector.push_back(mr);
    return mr;
  }
 
  
  // bool poll_completion_wid(uint64_t wid)
  // {
  //   if(_complete_ctr >= wid) return true;
    
  //   /* poll for completion of op */
  //   struct ibv_exp_wc wc[64];
  //   int ne = ibv_exp_poll_cq(_ctx->cq, 64, wc, sizeof(wc[0]));

  //   for (int i = 0; i < ne; ++i) {
  //     if (wc[i].status != IBV_WC_SUCCESS) {
  //       throw General_exception("Failed status (%d) for wr_id %d",
  //                               ibv_wc_status_str(wc[i].status),
  //                               wc[i].status, (int) wc[i].wr_id);
  //       break;
  //     }
  //     if(wc[i].wr_id) {
  //       _outstanding--;

  //       if(wc[i].wr_id >  _complete_ctr)
  //         _complete_ctr = wc[i].wr_id;
  //     }
  //   }

  //   if(ne==64) return poll_completion_wid(wid);

  //   return _complete_ctr >= wid;
  // }

  /** 
   * Poll completions and return number serviced
   * 
   * 
   * @return Number of completions serviced
   */
  int poll_completions(std::function<void(uint64_t)> release_func)
  {
    /* poll for completion of op */
    struct ibv_exp_wc wc[64];

    int ne = ibv_exp_poll_cq(_ctx->cq, 64, wc, sizeof(wc[0]));

    for (int i = 0; i < ne; ++i) {
      if (wc[i].status != IBV_WC_SUCCESS) {
        throw General_exception("Failed status %s (%d) for wr_id %d",
                                ibv_wc_status_str(wc[i].status),
                                wc[i].status, (int) wc[i].wr_id);
        break;
      }
      if(wc[i].wr_id) {
        _outstanding--;
        if(release_func)
          release_func(wc[i].wr_id);
      }
    }

    if(ne==64)
      return (poll_completions(release_func)+64);

    return ne;
  }


  /** 
   * Post send with work id.
   * 
   * @param first_sge_len 
   * @param wid 
   * @param extra_mr 
   * 
   * @return 
   */
  status_t post_send(uint64_t wid, struct ibv_mr * mr0, struct ibv_mr * extra_mr)
  {
    int rc = channel_post_send(_ctx, wid, mr0, extra_mr);
    if(rc) {
      PWRN("channel_post_send: errno=%d", rc);
      assert(0);
      return E_FAIL;
    }
    _outstanding++;
    return S_OK;
  }

  /** 
   * Post receive
   * 
   * @param wid 
   * @param mr 
   * 
   * @return 
   */
  status_t post_recv(uint64_t wid, struct ibv_mr * mr)
  {
    int rc = channel_post_recv(_ctx, wid, mr);
    if(rc) {
      assert(0);
      return E_FAIL;
    }
    _outstanding++;
    return S_OK;
  }

  inline int outstanding() const { return _outstanding; }

private:
  
  struct ibv_device * open_device(const char * device_name)
  {
    int num_devices = 0;
    struct ibv_device ** devices = ibv_get_device_list(&num_devices);
    struct ibv_device * result = nullptr;

    if(option_DEBUG)
      PLOG("detected %d ibv devices.", num_devices);

    if(num_devices == 0) throw Constructor_exception("no Rdma devices");

    if(option_DEBUG)
      for(int i=0;i<num_devices;i++)
        PLOG("  device:%s", ibv_get_device_name(devices[i]));

    if(device_name) {
      for(int i=0;i<num_devices;i++) {
        if(strcmp(ibv_get_device_name(devices[i]), device_name)==0) {
          if(option_DEBUG)
            PLOG("found device: %s", device_name);
          result = devices[i];
        }
      }    
    }
    else {
      if(option_DEBUG)
        PLOG("using first IB device: %s", ibv_get_device_name(devices[0]));
      result = devices[0];
    }

    if(!result)
      throw General_exception("rdma_transport failed to open device (%s)", device_name);

    ibv_free_device_list(devices);
    return result;
  }


  
private:
  struct ibv_device *          _device;
  struct channel_context *     _ctx;
  struct channel_dest          _local_endpoint;
  struct channel_dest *        _remote_endpoint;
  std::vector<struct ibv_mr *> _mr_vector; /**< registered memory regions */  
  char                         _gid[INET6_ADDRSTRLEN];
  int                          _outstanding; /**< outstanding requests */
  uint64_t _complete_ctr;
};


#endif // __COMANCHE_RDMA_TRANSPORT_H__

