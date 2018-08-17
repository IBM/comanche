/*
 * Copyright (c) 2005 Topspin Communications.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/** 
 * Based on code from TopSpin.  Daniel G. Waddington (C) IBM, 2017
 * 
 */
#include <common/logging.h>
#include <common/utils.h>
#include <common/exceptions.h>
#include "rdma_helper.h"

#define mmin(a, b) a < b ? a : b
#define MAX_SGE_LEN 0xFFFFFFF

static void *contig_addr = NULL;
  
int channel_connect_ctx(struct channel_context *ctx,
                        int port, int my_psn,
                        enum ibv_mtu mtu, int sl,
                        struct channel_dest *dest, int sgid_idx)
{
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof attr);
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = mtu;
  attr.dest_qp_num = dest->qpn;
  attr.rq_psn = dest->psn;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.is_global = 0;
  attr.ah_attr.dlid = dest->lid;
  attr.ah_attr.sl = sl;
  attr.ah_attr.src_path_bits = 0;
  attr.ah_attr.port_num = port;

  if (dest->gid.global.interface_id) {
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.hop_limit = 1;
    attr.ah_attr.grh.dgid = dest->gid;
    attr.ah_attr.grh.sgid_index = sgid_idx;
  }

  if(ctx->qtype == IBV_QPT_RC) {
    if (ibv_modify_qp(ctx->qp, &attr,
                      IBV_QP_STATE              |
                      IBV_QP_AV                 |
                      IBV_QP_PATH_MTU           |
                      IBV_QP_DEST_QPN           |
                      IBV_QP_RQ_PSN             |
                      IBV_QP_MAX_DEST_RD_ATOMIC |
                      IBV_QP_MIN_RNR_TIMER)) {
      PERR("Failed to modify QP to RTR");
      return 1;
    }

    attr.qp_state	    = IBV_QPS_RTS;
    attr.timeout	    = 14;
    attr.retry_cnt	    = 10;
    attr.rnr_retry	    = 10;
    attr.sq_psn	    = my_psn;
    attr.max_rd_atomic  = 1;

    if (ibv_modify_qp(ctx->qp, &attr,
                      IBV_QP_STATE              |
                      IBV_QP_TIMEOUT            |
                      IBV_QP_RETRY_CNT          |
                      IBV_QP_RNR_RETRY          |
                      IBV_QP_SQ_PSN             |
                      IBV_QP_MAX_QP_RD_ATOMIC)) {
      PERR("Failed to modify QP to RTS");
      return 1;
    }
  }
  else if(ctx->qtype == IBV_QPT_UC) {
    if (ibv_modify_qp(ctx->qp, &attr,
                      IBV_QP_STATE              |
                      IBV_QP_AV                 |
                      IBV_QP_PATH_MTU           |
                      IBV_QP_DEST_QPN           |
                      IBV_QP_RQ_PSN)) {
      PERR("Failed to modify QP to RTR");
      return 1;
    }

    attr.qp_state	    = IBV_QPS_RTS;
    attr.sq_psn	    = my_psn;

    if (ibv_modify_qp(ctx->qp, &attr,
                      IBV_QP_STATE              |
                      IBV_QP_SQ_PSN)) {
      PERR("Failed to modify QP to RTS");
      return 1;
    }
  }
  else {
    assert(0);
    return 1;
  }

  return 0;
}

struct channel_dest *channel_client_exch_dest(const char *servername,
                                              int port,
                                              const struct channel_dest *my_dest)
{
  struct addrinfo *res, *t;

  struct addrinfo hints = {0};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  
  char service[255];
  char msg[sizeof "0000:000000:000000:00000000000000000000000000000000"];
  int n;
  int sockfd = -1;
  struct channel_dest *rem_dest = NULL;
  char gid[33];

  if (sprintf(service, "%d", port) < 0)
    return NULL;

  n = getaddrinfo(servername, service, &hints, &res);

  if (n < 0) {
    PERR("%s for %s:%d\n", gai_strerror(n), servername, port);
    return NULL;
  }

  for (t = res; t; t = t->ai_next) {
    sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
    if (sockfd >= 0) {
      if (!connect(sockfd, t->ai_addr, t->ai_addrlen))
	break;
      close(sockfd);
      sockfd = -1;
    }
  }

  freeaddrinfo(res);

  if (sockfd < 0) {
    return NULL;
  }

  gid_to_wire_gid(&my_dest->gid, gid);
  sprintf(msg, "%04x:%06x:%06x:%s", my_dest->lid, my_dest->qpn,
	  my_dest->psn, gid);
  if (write(sockfd, msg, sizeof msg) != sizeof msg) {
    PERR("Couldn't send local address");
    goto out;
  }

  if (recv(sockfd, msg, sizeof(msg), MSG_WAITALL) != sizeof(msg)) {
    PERR("Couldn't read remote address\n");
    goto out;
  }

  if (write(sockfd, "done", sizeof("done")) != sizeof("done")) {
    PERR("Couldn't send \"done\" msg\n");
    goto out;
  }

  rem_dest = (channel_dest *) malloc(sizeof *rem_dest);
  if (!rem_dest)
    goto out;

  sscanf(msg, "%x:%x:%x:%s", &rem_dest->lid, &rem_dest->qpn,
	 &rem_dest->psn, gid);
  wire_gid_to_gid(gid, &rem_dest->gid);

 out:
  close(sockfd);
  return rem_dest;
}

struct channel_dest *channel_server_exch_dest(struct channel_context *ctx,
					      int ib_port, enum ibv_mtu mtu,
					      int port, int sl,
					      const struct channel_dest *my_dest,
					      int sgid_idx)
{
  struct addrinfo *res, *t;

  struct addrinfo hints = {0};
  hints.ai_flags = AI_PASSIVE;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  char service[255];
  char msg[sizeof "0000:000000:000000:00000000000000000000000000000000"];
  int n;
  int sockfd = -1, connfd;
  struct channel_dest *rem_dest = NULL;
  char gid[33];

  if (sprintf(service, "%d", port) < 0)
    return NULL;

  n = getaddrinfo(NULL, service, &hints, &res);

  if (n < 0) {
    PERR("%s for port %d", gai_strerror(n), port);
    return NULL;
  }

  for (t = res; t; t = t->ai_next) {
    sockfd = socket(t->ai_family, t->ai_socktype, t->ai_protocol);
    if (sockfd >= 0) {
      n = 1;

      setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &n, sizeof n);

      if (!bind(sockfd, t->ai_addr, t->ai_addrlen))
	break;
      close(sockfd);
      sockfd = -1;
    }
  }

  freeaddrinfo(res);

  if (sockfd < 0) {
    PERR("Couldn't listen to port %d", port);
    return NULL;
  }

  listen(sockfd, 1);
  connfd = accept(sockfd, NULL, 0);
  close(sockfd);
  if (connfd < 0) {
    fprintf(stderr, "accept() failed\n");
    return NULL;
  }

  n = recv(connfd, msg, sizeof(msg), MSG_WAITALL);
  if (n != sizeof msg) {
    perror("server read");
    fprintf(stderr, "%d/%d: Couldn't read remote address\n", n, (int) sizeof msg);
    goto out;
  }

  rem_dest = (channel_dest *) malloc(sizeof *rem_dest);
  if (!rem_dest)
    goto out;

  sscanf(msg, "%x:%x:%x:%s", &rem_dest->lid, &rem_dest->qpn,
	 &rem_dest->psn, gid);
  wire_gid_to_gid(gid, &rem_dest->gid);

  if (channel_connect_ctx(ctx, ib_port, my_dest->psn, mtu, sl, rem_dest,
			  sgid_idx)) {
    PERR("Couldn't connect to remote QP");
    free(rem_dest);
    rem_dest = NULL;
    goto out;
  }


  gid_to_wire_gid(&my_dest->gid, gid);
  sprintf(msg, "%04x:%06x:%06x:%s", my_dest->lid, my_dest->qpn,
	  my_dest->psn, gid);
  if (write(connfd, msg, sizeof msg) != sizeof msg) {
    PERR("Couldn't send local address");
    free(rem_dest);
    rem_dest = NULL;
    goto out;
  }

  /* expecting "done" msg */
  if (read(connfd, msg, sizeof(msg)) <= 0) {
    PERR("Couldn't read \"done\" msg");
    free(rem_dest);
    rem_dest = NULL;
    goto out;
  }

 out:
  close(connfd);
  return rem_dest;
}

struct channel_context *channel_init_ctx(struct ibv_device *ib_dev,
                                         enum ibv_qp_type qtype,
                                         unsigned long long size,
                                         int rx_depth,
                                         int tx_depth,
                                         int port,
                                         int use_event,
                                         int inlr_recv
                                         )
{
  struct channel_context *ctx;
  struct ibv_exp_device_attr dattr;
  int ret;

  ctx = (struct channel_context*) calloc(1, sizeof *ctx);
  if (!ctx)
    return NULL;

  ctx->qtype = qtype;
  assert(qtype == IBV_QPT_RC || qtype == IBV_QPT_UC);

  memset(&dattr, 0, sizeof(dattr));

  //  ctx->buffers.push_back({NULL, memalign(buffer_size, size), buffer_size});
  ctx->size = size;
  ctx->rx_depth = rx_depth;

  ctx->context = ibv_open_device(ib_dev);
  if (!ctx->context) {
    PERR("Couldn't get context for %s",
	 ibv_get_device_name(ib_dev));
    goto clean_device;
  }

  if (inlr_recv) {
    PLOG("Using inline receive");
    dattr.comp_mask |= IBV_EXP_DEVICE_ATTR_INLINE_RECV_SZ;
    ret = ibv_exp_query_device(ctx->context, &dattr);
    if (ret) {
      PERR("  Couldn't query device for inline-receive capabilities.\n");
    } else if (!(dattr.comp_mask & IBV_EXP_DEVICE_ATTR_INLINE_RECV_SZ)) {
      PERR("  Inline-receive not supported by driver.\n");
    } else if (dattr.inline_recv_sz < inlr_recv) {
      PERR("  Max inline-receive(%d) < Requested inline-receive(%d).\n",
	   dattr.inline_recv_sz, inlr_recv);
    }

    PLOG("Max QP: %d", dattr.max_qp);
    PLOG("Max QP WR: %d", dattr.max_qp_wr);
  }
  ctx->inlr_recv = inlr_recv;

  if (use_event) {
    ctx->channel = ibv_create_comp_channel(ctx->context);
    if (!ctx->channel) {
      PERR("Couldn't create completion channel\n");
      goto clean_device;
    }
  }
  else {
    ctx->channel = NULL;
  }

  ctx->pd = ibv_alloc_pd(ctx->context);
  if (!ctx->pd) {
    PERR("Couldn't allocate PD");
    goto clean_comp_channel;
  }

  {
    struct ibv_exp_reg_mr_in in = {0};
    in.pd = ctx->pd;
    in.addr = contig_addr;
    in.length = size;
    in.exp_access = IBV_EXP_ACCESS_LOCAL_WRITE;
    if (contig_addr) {
      in.comp_mask = IBV_EXP_REG_MR_CREATE_FLAGS;
      in.create_flags = IBV_EXP_REG_MR_CREATE_CONTIG;
    } else {
      in.comp_mask = 0;
      in.exp_access |= IBV_EXP_ACCESS_ALLOCATE_MR;
    }

    //    ctx->mr = ibv_exp_reg_mr(&in);
  }
		
  //  if (!ctx->mr) {
  //  PERR("Couldn't register MR");
  //  goto clean_pd;
  // }
	
  //  ctx->buf = ctx->mr->addr;

  /*FIXME memset(ctx->buf, 0, size); */
  //  memset(ctx->buf, 0x0, size);

  ctx->cq = ibv_create_cq(ctx->context,
                          rx_depth + 1,
                          NULL,
                          ctx->channel, 0);
  if (!ctx->cq) {
    PERR("Couldn't create CQ");
    goto clean_mr;
  }

  /* create QP */
  {
    assert(ctx->cq);

    struct ibv_exp_qp_init_attr attr = {0};

    attr.send_cq = ctx->cq;
    attr.recv_cq = ctx->cq;
    attr.cap.max_send_wr = tx_depth;
    attr.cap.max_recv_wr = rx_depth;
    attr.cap.max_send_sge = 2;
    attr.cap.max_recv_sge = 2;
    attr.qp_type = ctx->qtype; // IBV_QPT_RC or IBV_QPT_UC    
    attr.pd = ctx->pd;
    attr.comp_mask = IBV_EXP_QP_INIT_ATTR_PD;
    attr.max_inl_recv = ctx->inlr_recv;
      
    if (ctx->inlr_recv)
      attr.comp_mask |= IBV_EXP_QP_INIT_ATTR_INL_RECV;

    ctx->qp = ibv_exp_create_qp(ctx->context, &attr);

    if (!ctx->qp)  {
      PERR("Couldn't create QP");
      goto clean_cq;
    }
    PLOG("Max inline recv: %u", attr.max_inl_recv);
    // if (ctx->inlr_recv > attr.max_inl_recv)
    // 	PLOG("  Actual inline-receive(%d) < requested inline-receive(%d)",
    //        attr.max_inl_recv, ctx->inlr_recv);
  }
  assert(ctx->qp);
  PLOG("QP creation OK");
  
  {
    struct ibv_qp_attr attr;
    memset(&attr,0,sizeof attr);
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = port;
    attr.qp_access_flags = 0;
    
    if (ibv_modify_qp(ctx->qp, &attr,
		      IBV_QP_STATE              |
		      IBV_QP_PKEY_INDEX         |
		      IBV_QP_PORT               |
		      IBV_QP_ACCESS_FLAGS)) {
      PERR("Failed to modify QP to INIT errno=%d", errno);
      goto clean_qp;
    }
  }

  return ctx;

 clean_qp:
  ibv_destroy_qp(ctx->qp);

 clean_cq:
  ibv_destroy_cq(ctx->cq);

 clean_mr:
  //  ibv_dereg_mr(ctx->mr);

  //clean_pd:
  ibv_dealloc_pd(ctx->pd);

 clean_comp_channel:
  if (ctx->channel)
    ibv_destroy_comp_channel(ctx->channel);

 clean_device:
  ibv_close_device(ctx->context);
  free(ctx);

  return NULL;
}


struct ibv_mr * channel_register_memory(struct channel_context *ctx,
                                        void * contig_addr,
                                        size_t len)
{
  //  PLOG("channel_register_memory: ctx=%p contig_addr=%p, len=%lu", ctx, contig_addr, len);  
  assert(ctx);
  assert(contig_addr);
  
  struct ibv_mr * mr = ibv_reg_mr(ctx->pd,
                                  contig_addr,
                                  len,

                                  IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  assert(mr);
  //  PDBG("--- post registered mr = %p", (void *) mr);
  return mr;
}


int channel_close_ctx(struct channel_context *ctx)
{
  if(!ctx) return 0;
  
  if (ibv_destroy_qp(ctx->qp)) {
    PERR("Couldn't destroy QP\n");
    return 1;
  }

  if (ibv_destroy_cq(ctx->cq)) {
    PERR("Couldn't destroy CQ\n");
    return 1;
  }

  // if (ibv_dereg_mr(ctx->mr)) {
  //   PERR("Couldn't deregister MR\n");
  //   return 1;
  // }

  if (ibv_dealloc_pd(ctx->pd)) {
    PERR("Couldn't deallocate PD\n");
    return 1;
  }

  if (ctx->channel) {
    if (ibv_destroy_comp_channel(ctx->channel)) {
      PERR("Couldn't destroy completion channel\n");
      return 1;
    }
  }

  if (ibv_close_device(ctx->context)) {
    PERR("Couldn't release context\n");
    return 1;
  }


  free(ctx);

  return 0;
}


int channel_post_recv(struct channel_context *ctx, uint64_t issue_id, struct ibv_mr * mr)
{
  assert(mr);
  
  struct ibv_sge list = {0};
  list.addr = (uint64_t) mr->addr;
  list.length = mr->length;
  list.lkey = mr->lkey;

  struct ibv_recv_wr wr = {0};
  wr.wr_id	 = issue_id; //CHANNEL_RECV_WRID;
  wr.sg_list = &list;
  wr.num_sge = 1;
  
  struct ibv_recv_wr *bad_wr;
  int rc;
  if ((rc=ibv_post_recv(ctx->qp, &wr, &bad_wr))) {
    throw General_exception("ibv_post_recv failed! errno=%d", errno);
  }
  return rc;
}


// int channel_post_recv(struct channel_context *ctx,
//                       uint64_t issue_id,
//                       size_t first_sge_len,
//                       int n,
//                       struct ibv_mr * extra_mr)
// {
//   int i;
  
//   if(extra_mr) {
    
//     struct ibv_sge list[2] = {0};
//     list[0].addr = (uint64_t) ctx->buf;
//     list[0].length = first_sge_len;
//     list[0].lkey = ctx->mr->lkey;
//     list[1].addr = (uint64_t) extra_mr->addr;
//     list[1].length = extra_mr->length;
//     list[1].lkey = extra_mr->lkey;
//     PLOG("post_recv extra mr: addr=%p len=%lu", extra_mr->addr, extra_mr->length);
    
//     struct ibv_recv_wr wr = {0};
//     wr.wr_id	 = issue_id; //CHANNEL_RECV_WRID;
//     wr.sg_list = list;
//     wr.num_sge = 2;
  
//     struct ibv_recv_wr *bad_wr;

//     for (i = 0; i < n; ++i) {
//       if (ibv_post_recv(ctx->qp, &wr, &bad_wr)) {
//         throw General_exception("ibv_post_recv (with extra mr) failed! errno=%d", errno);
//         break;
//       }
//     }

//   }
//   else {
    
//     struct ibv_sge list = {0};
//     list.addr = (uint64_t) ctx->buf;
//     list.length = mmin(ctx->size, MAX_SGE_LEN);
//     list.lkey = ctx->mr->lkey;

//     struct ibv_recv_wr wr = {0};
//     wr.wr_id	 = CHANNEL_RECV_WRID;
//     wr.sg_list = &list;
//     wr.num_sge = 1;
  
//     struct ibv_recv_wr *bad_wr;

//     for (i = 0; i < n; ++i) {
//       if (ibv_post_recv(ctx->qp, &wr, &bad_wr)) {
//         throw General_exception("ibv_post_recv failed! errno=%d", errno);
//         break;
//       }
//     }

//   }

//   return i;
// }

int channel_post_send(struct channel_context *ctx,
                      uint64_t issue_id,
                      struct ibv_mr * mr0,
                      struct ibv_mr * extra_mr)
{
  if(extra_mr == nullptr) {
    struct ibv_sge seg = {0};
    seg.addr = (uintptr_t) mr0->addr;
    seg.length = mr0->length;
    seg.lkey = mr0->lkey;
    
    struct ibv_send_wr wr = {0};
    wr.wr_id = issue_id;
    wr.sg_list = &seg;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad_wr;
    return ibv_post_send(ctx->qp, &wr, &bad_wr);    
  }
  else {
    struct ibv_sge list[2] = {0};
    list[0].addr = (uintptr_t) mr0->addr;
    list[0].length = mr0->length;
    list[0].lkey = mr0->lkey;
    list[1].addr = (uintptr_t) extra_mr->addr;
    list[1].length = extra_mr->length;
    list[1].lkey = extra_mr->lkey;
    
    struct ibv_send_wr wr = {0};
    wr.wr_id	 = issue_id;
    wr.sg_list = list;
    wr.num_sge = 2;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
  
    struct ibv_send_wr *bad_wr;
    return ibv_post_send(ctx->qp, &wr, &bad_wr);
  }

}


// int channel_post_send_trim(struct channel_context *ctx,
// 			   uint64_t issue_id,
// 			   size_t first_sge_len,
// 			   struct ibv_mr * extra_mr,                      
// 			   size_t offset,
// 			   size_t length)
// {
    
//   if(extra_mr) {
//     assert(first_sge_len);    
//     assert(first_sge_len + extra_mr->length <= ctx->size);
//     struct ibv_sge list[2] = {0};
//     list[0].addr = (uintptr_t) ctx->buf;
//     list[0].length = first_sge_len;
//     list[0].lkey = ctx->mr->lkey;

//     list[1].addr = (uintptr_t) extra_mr->addr;

//     if(offset) {
//       list[1].addr += offset;
//       if(length) {
//         list[1].length = length;
//         assert((offset + length) <= extra_mr->length);
//       }
//       else {
//         list[1].length = extra_mr->length - offset;
//       }
//     }
//     else {
//       list[1].length = extra_mr->length;
//     }
//     list[1].lkey = extra_mr->lkey;

//     //    PLOG("extra mr: addr=%p len=%lu", extra_mr->addr, extra_mr->length);
  
//     struct ibv_send_wr wr = {0};
//     wr.wr_id = issue_id; //CHANNEL_SEND_WRID;
//     wr.sg_list = list;
//     wr.num_sge = 2;
//     wr.opcode = IBV_WR_SEND;
//     wr.send_flags = IBV_SEND_SIGNALED;

//     struct ibv_send_wr *bad_wr;
    
//     return ibv_post_send(ctx->qp, &wr, &bad_wr);
//   }
//   else {
//     struct ibv_sge list = {0};
//     list.addr = (uintptr_t) ctx->buf;
//     list.length = first_sge_len > 0 ? first_sge_len : ctx->size;
//     list.lkey = ctx->mr->lkey;
    
//     struct ibv_send_wr wr = {0};
//     wr.wr_id = issue_id; 
//     wr.sg_list = &list;
//     wr.num_sge = 1;
//     wr.opcode = IBV_WR_SEND;
//     wr.send_flags = IBV_SEND_SIGNALED;

//     struct ibv_send_wr *bad_wr;

//     return ibv_post_send(ctx->qp, &wr, &bad_wr);
//   }
// }


enum ibv_mtu rdma_mtu_to_enum(int mtu)
{
  switch (mtu) {
  case 256:  return IBV_MTU_256;
  case 512:  return IBV_MTU_512;
  case 1024: return IBV_MTU_1024;
  case 2048: return IBV_MTU_2048;
  case 4096: return IBV_MTU_4096;
  default:   assert(0);
  }
  return IBV_MTU_4096;
}

uint16_t rdma_get_local_lid(struct ibv_context *context, int port)
{
  struct ibv_port_attr attr;

  if (ibv_query_port(context, port, &attr))
    return 0;

  return attr.lid;
}

int rdma_get_port_info(struct ibv_context *context, int port,
                       struct ibv_port_attr *attr)
{
  return ibv_query_port(context, port, attr);
}

void wire_gid_to_gid(const char *wgid, union ibv_gid *gid)
{
  char tmp[9];
  uint32_t v32;
  uint32_t *raw = (uint32_t *)gid->raw;
  int i;

  for (tmp[8] = 0, i = 0; i < 4; ++i) {
    memcpy(tmp, wgid + i * 8, 8);
    sscanf(tmp, "%x", &v32);
    raw[i] = ntohl(v32);
  }
}

void gid_to_wire_gid(const union ibv_gid *gid, char wgid[])
{
  int i;
  uint32_t *raw = (uint32_t *)gid->raw;

  for (i = 0; i < 4; ++i) {
    sprintf(&wgid[i * 8], "%08x",htonl(raw[i]));
  }
}


