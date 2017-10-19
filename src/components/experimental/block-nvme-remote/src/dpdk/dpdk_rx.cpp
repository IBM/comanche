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
#include "config.h"

#include <common/logging.h>
#include <common/cycles.h>
#include <common/utils.h>
#include <assert.h>
#include <unistd.h>
#include <rte_ethdev.h>
#include <rte_arp.h>
#include <rte_mempool.h>
#include <rte_ethdev.h>
#include <rte_lcore.h>

#include "nvme_buffer.h"
#include "dpdk_eth.h"
#include "eal_init.h"

#include <lwip/init.h>
#include <lwip/etharp.h>
#include <lwip/timeouts.h>
#include <lwip/tcpip.h>
#include <lwip/tcp.h>
#include <lwip/ip.h>
#include <lwip/def.h>
#include <lwip/mem.h>
#include <lwip/ip4.h>
#include <lwip/ip4_frag.h>
#include <lwip/inet_chksum.h>
#include <lwip/netif.h>
#include <lwip/sockets.h>
#include <netif/ethernet.h>


/*--------------------------------------------------------
 * Rx Queue class implementation
 *--------------------------------------------------------
 */

DPDK::Ethernet_rx_queue::
Ethernet_rx_queue(Ethernet_device& device,
                  unsigned queue_id,
                  unsigned lcore_id,
                  Ethernet_tx_queue * tx_queue) :
  _device(device),
  _queue_id(queue_id),
  _lcore_id(lcore_id),
  _exit_flag(false),
  _tx_queue(tx_queue)
{  
  /* create per-queue RX mbuf pools */
  char buffer_id[32];
  sprintf(buffer_id,"mbuf-pool-rx-%u",queue_id);
  _rx_mbuf_pool = rte_pktmbuf_pool_create(buffer_id,
                                          NB_MBUF,
                                          MEMPOOL_CACHE_SIZE,
                                          0,
                                          RTE_MBUF_DEFAULT_BUF_SIZE,
                                          rte_eth_dev_socket_id(device.port_id()));

  int rc = rte_eth_rx_queue_setup(device.port_id(),
                                  queue_id,
                                  NB_RXDESC,
                                  rte_eth_dev_socket_id(device.port_id()),
                                  NULL, // rxconf
                                  _rx_mbuf_pool);
  if(rc)
    throw General_exception("rte_eth_rx_queue_setup failed");

  PLOG("RxQueue (%u) created.", queue_id);

  /* launch thread */
  rc = rte_eal_remote_launch(lcore_entry_trampoline, this, lcore_id);
  if(rc)
    throw General_exception("rte_eal_remote_launch failed for Rx queue (%u)", _queue_id);
  
}

DPDK::Ethernet_rx_queue::
~Ethernet_rx_queue()
{
  _exit_flag = true;
  rte_eal_wait_lcore(_lcore_id); /* join worker */  
}

struct netif * DPDK::Ethernet_rx_queue::netif() const {
  return _device._netif;
}


int DPDK::Ethernet_rx_queue::lcore_entry(void * arg)
{
  assert(arg);
  const unsigned BURST_SIZE = 5;
  struct rte_mbuf * p_rx_buffers[BURST_SIZE];
  uint8_t port_id = _device.port_id();
  static unsigned long proto_packets = 0;
    
  while(!_exit_flag) {


    if(_queue_id == 0) {
      sys_check_timeouts();
    }
    
    uint16_t rc;

    rc = rte_eth_rx_burst(port_id,
                          _queue_id,
                          p_rx_buffers,
                          BURST_SIZE);
    //    PLOG("rte_eth_rx_burst return code:%u",rc);
      
    
    for(unsigned p=0;p<rc;p++) {

      struct rte_mbuf * rtebuf = p_rx_buffers[p];
      struct ether_hdr *eth_hdr = rte_pktmbuf_mtod(rtebuf,
                                                   struct ether_hdr *);
      
      char src_addr[64];
      ether_format_addr(src_addr, 64, &eth_hdr->s_addr);
      
      char dst_addr[64];
      ether_format_addr(dst_addr, 64, &eth_hdr->d_addr);

#if 0
      unsigned char * payload = ((unsigned char*)p_rx_buffers[p]) + sizeof(struct ether_hdr);
      PINF("Rx packet [%u]: %s - %s [%.4x]: %.2x %.2x %.2x %.2x %.2x %.2x",
           _queue_id,
           dst_addr, src_addr, rte_cpu_to_be_16(eth_hdr->ether_type),
           payload[0],payload[1],payload[2],payload[3],payload[4],payload[5]);
#endif

      /* get hold of ip_hdr */
      struct ip_hdr *ip_hdr = rte_pktmbuf_mtod_offset(rtebuf,
                                                      struct ip_hdr *,
                                                      sizeof(struct ether_hdr));

      struct udp_hdr *udp_hdr = rte_pktmbuf_mtod_offset(rtebuf,
                                                        struct udp_hdr *,
                                                        sizeof(struct ether_hdr)+sizeof(struct ip_hdr));

      char * payload = rte_pktmbuf_mtod_offset(rtebuf,
                                               char*,
                                               sizeof(struct udp_hdr)+sizeof(struct ether_hdr)+sizeof(struct ip_hdr));

      /* check for protocol packets */
      if(IPH_PROTO(ip_hdr) == IP_PROTO_UDP) {
        if(udp_hdr->dst_port == lwip_htons(10000)) {
          //PNOTICE("Found protocol packet!");
          /* for the moment bounce packet */
          _tx_queue->bounce_packet(rtebuf);
          proto_packets++;
          if(proto_packets % 100 == 0){
            PNOTICE("Received %lu proto packets", proto_packets);
          }
        }
        else if(udp_hdr->dst_port == lwip_htons(100)) {
          cpu_time_t start =  *((uint64_t*)payload);
          PNOTICE("Got response! rtt=%f", ((float)(rdtsc() - start))/2400.0);
          _received++;
        }
        else {
          PLOG("unknown UDP port %d",udp_hdr->dst_port);
        }
        continue;
      }
      
      /* otherwise upcall stack ... */

      /* translate buffer from DPDK to LWIP */
      struct pbuf_custom pb_cust;      
      pb_cust.custom_free_function = [](struct pbuf *p) {};   

      assert(rtebuf->nb_segs == 1);
      auto pb = pbuf_alloced_custom(PBUF_RAW,
                                    rtebuf->pkt_len,
                                    PBUF_RAM,
                                    &pb_cust,
                                    (void*) eth_hdr,
                                    rtebuf->pkt_len);
      
      /* call into LWIP stack */
      ethernet_input(pb,_device._netif);

    }

    /* Free buffers. */
    for(unsigned p=0;p<rc;p++)
      rte_pktmbuf_free(p_rx_buffers[p]);
  }

  assert(_exit_flag);
  return 0;
}



unsigned DPDK::Ethernet_rx_queue::_received = 0;
