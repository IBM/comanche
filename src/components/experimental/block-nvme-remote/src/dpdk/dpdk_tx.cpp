#include <rte_malloc.h>
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

#include "dpdk_eth.h"
#include "config.h"

/*--------------------------------------------------------
 * Tx Queue class implementation
 *--------------------------------------------------------
 */
DPDK::Ethernet_tx_queue::
Ethernet_tx_queue(Ethernet_device& device,
                  unsigned queue_id) :
  _device(device),
  _queue_id(queue_id)
{
  char buffer_id[32];
  sprintf(buffer_id, "mbuf-pool-tx-%u", queue_id);
  
  _tx_mbuf_pool = rte_pktmbuf_pool_create(buffer_id,
                                          NB_MBUF,
                                          MEMPOOL_CACHE_SIZE,
                                          0,
                                          RTE_MBUF_DEFAULT_BUF_SIZE,
                                          rte_socket_id());
  assert(_tx_mbuf_pool);

  sprintf(buffer_id,"buffer-tx--%u", queue_id);
  _tx_buffer = (struct rte_eth_dev_tx_buffer *) rte_zmalloc_socket(buffer_id,
                                                                   RTE_ETH_TX_BUFFER_SIZE(MAX_PKT_BURST), 0,
                                                                   rte_socket_id());
  assert(_tx_buffer);
  rte_eth_tx_buffer_init(_tx_buffer, MAX_PKT_BURST);


  /* TX queue configuration, e.g., offloading */
  memset(&_tx_conf, 0, sizeof(_tx_conf));
  _tx_conf.txq_flags = 0;

  int rc = rte_eth_tx_queue_setup(device.port_id(),
                                  queue_id,
                                  NB_TXDESC,
                                  rte_eth_dev_socket_id(device.port_id()),
                                  &_tx_conf);

  if(rc)
    throw General_exception("rte_eth_tx_queue_setup failed");

  PLOG("TxQueue (%u) created.", queue_id);
}

DPDK::Ethernet_tx_queue::
~Ethernet_tx_queue()
{
  rte_free(_tx_buffer);
  rte_mempool_free(_tx_mbuf_pool);
}


struct rte_mbuf *
DPDK::Ethernet_tx_queue::
allocate_packet()
{
  return rte_pktmbuf_alloc(_tx_mbuf_pool);
}

struct rte_mbuf *
DPDK::Ethernet_tx_queue::
clone_packet(struct rte_mbuf *mb)
{
  return rte_pktmbuf_clone(mb, _tx_mbuf_pool);
}



void
DPDK::Ethernet_tx_queue::
release_packet(struct rte_mbuf* pkt)
{
  assert(pkt);
  rte_pktmbuf_free(pkt);
}


static struct rte_mbuf * create_fake_packet(void * payload,
                                     size_t payload_len,
                                     uint8_t port_id)
{
  struct rte_mbuf * fake = (struct rte_mbuf *) rte_zmalloc("fakembuf",sizeof(struct rte_mbuf),2048);
  if(!fake)
    throw General_exception("rte_malloc: failed in __create_fake_packet");
  
  /* NOTE: this fake packet won't be freed by the driver 
     see __rte_pktmbuf_prefree_seg(struct rte_mbuf *m) 
    */  
  rte_pktmbuf_reset(fake);
  fake->nb_segs = 1;
  fake->port = port_id;
  fake->buf_addr = payload;
  fake->buf_physaddr = rte_malloc_virt2phy(payload);
  fake->buf_len = payload_len;
  fake->data_len = payload_len;
  fake->data_off = 0;
  fake->refcnt = 0; /* this will cause no release */

  return fake;
}

int
DPDK::Ethernet_tx_queue::
ethernet_send_burst_raw(struct ether_addr& dst_addr,
                        uint16_t eth_type,
                        size_t burst_size,
                        struct rte_mbuf **packets,
                        void ** payload,
                        size_t * payload_len)
{
  if(burst_size > MAX_PKT_BURST)
    throw General_exception("burst size too big");

  uint16_t transmit_count = 0;
  
  /* prepare all of the packets in a burst */
  for(size_t i=0;i<burst_size;i++) {

    PLOG("preparing packet %ld", i);
    struct rte_mbuf * pkt = packets[i];

    if(!pkt)
      throw General_exception("bad param in ethernet_send_burst_raw");
    
    struct ether_hdr *eth_hdr = (struct ether_hdr *)
      rte_pktmbuf_prepend(pkt, (uint16_t)sizeof(struct ether_hdr));
    
    if(!eth_hdr)
      throw General_exception("ethernet_send: not enough head-room");

    ether_addr_copy(&_device._eth_addr, &eth_hdr->s_addr); // set source MAC address
    ether_addr_copy(&dst_addr, &eth_hdr->d_addr); // set dest MAC address
    eth_hdr->ether_type = rte_be_to_cpu_16(eth_type);

    assert(payload_len[i] > 0);
    assert(payload[i]);
    
#ifdef TARGET_MELLANOX
    char * payload_offset = rte_pktmbuf_append(pkt, payload_len[i]); /* append two bytes */
    assert(payload_offset);
    rte_memcpy(payload_offset, payload[i], payload_len[i]);
#else
    assert(0);
#endif


    /* Create a fake mbuf packet so we can build a packet from
       multiple segments. This won't belong to a pool so we have to
       catch it before it is released.
    */
    // struct rte_mbuf * fake = create_fake_packet(payload[i],
    //                                      payload_len[i],
    //                                      _device._port_id);
#if 0
    struct rte_mbuf * payload_packet = rte_pktmbuf_alloc(_device._mbuf_payload_pool[_queue_id]);
    assert(payload_packet);

    char * pl = rte_pktmbuf_append(payload_packet, payload_len[i]);
    memset((void*)pl,0xA,payload_len[i]);
    PLOG("payload packet len:%lu", rte_pktmbuf_data_len(payload_packet));
    
    //    fake->buf_addr = rte_malloc("fancy",4096,4096);
    //    fake->buf_physaddr = rte_malloc_virt2phy(fake->buf_addr);
    /* Chain buffer */
    if(rte_pktmbuf_chain(pkt, payload_packet)!=0)
      throw General_exception("buffer chaining failed");
#endif
  }

#if RTE_VER_YEAR >= 17
  if(rte_eth_tx_prepare(_device._port_id,
                        _queue_id,
                        packets,
                        burst_size) != burst_size)
    throw General_exception("rte_eth_tx_prepare: failed unexpectedly");
#endif

  /* perform burst TX */
  transmit_count = rte_eth_tx_burst(_device._port_id,
                                    _queue_id,
                                    packets,
                                    burst_size);

  PLOG("burst transmit count: %u", transmit_count);

  return transmit_count;
}

int
DPDK::Ethernet_tx_queue::
ethernet_send_frame_ex(struct rte_mbuf* header_segment,
                       void * frame,
                       size_t frame_len)
{
#ifdef DISABLE_FAKE
  /* create a packet */
  struct rte_mbuf * payload_mbuf = rte_pktmbuf_alloc(_tx_mbuf_pool);
  /* memory copy frame */
  char * fp = rte_pktmbuf_append(payload_mbuf, frame_len);
  rte_memcpy(fp, frame, frame_len);
#else
  struct rte_mbuf * payload_mbuf = create_fake_packet(frame, frame_len,_device._port_id);
#endif
  
  assert(payload_mbuf);
  
  if(rte_pktmbuf_chain(header_segment, payload_mbuf)!=0)
    throw General_exception("buffer chaining failed");

  /* send packet */
  int send_count = rte_eth_tx_buffer(_device._port_id,
                                     _queue_id,
                                     _tx_buffer,
                                     header_segment);
  
  /* synchronous flush for the moment */
  send_count += rte_eth_tx_buffer_flush(_device._port_id,
                                        _queue_id,
                                        _tx_buffer);

  if(send_count != 1)
    throw General_exception("rte_eth_tx failure");

  //  PLOG("ethernet_send_frame_ex: send count = %d", send_count);
  return send_count;
}

// static uint16_t rte_eth_tx_burst 	( 	uint8_t  	port_id,
// 		uint16_t  	queue_id,
// 		struct rte_mbuf **  	tx_pkts,
// 		uint16_t  	nb_pkts 
// 	)
  
int
DPDK::Ethernet_tx_queue::
ethernet_send_burst_ex(struct rte_mbuf* header_segment,
                       std::vector<payload_t>& pv)
{
  size_t burst_size = pv.size();
  assert(burst_size > 0);

  if(burst_size > Ethernet_tx_queue::MAX_PKT_BURST)
    throw General_exception("burst size too big");

  
  struct rte_mbuf ** tx_pkts = new struct rte_mbuf * [burst_size];

  int i=0;
  for(auto& p: pv) { /* iterate payload vector */
    
    /* create fake packet */
    struct rte_mbuf * payload_mbuf = create_fake_packet(p.data,
                                                        p.len,
                                                        _device._port_id);

    /* clone header then chain payload */
    tx_pkts[i] = clone_packet(header_segment);
    if(rte_pktmbuf_chain(tx_pkts[i], payload_mbuf)!=0)
      throw General_exception("buffer chaining failed");
    i++;
  }

  uint16_t sent = rte_eth_tx_burst(_device._port_id,
                                   _queue_id,
                                   tx_pkts,
                                   (uint16_t) burst_size);

  delete tx_pkts;
  
  return sent;
}


int
DPDK::Ethernet_tx_queue::
ethernet_send_frame(void * frame,
                    size_t frame_len)
{

  /* create a packet */
  struct rte_mbuf * packet = rte_pktmbuf_alloc(_tx_mbuf_pool);
  assert(packet);

  /* memory copy frame - this should only be used for ARP, ICMP etc. */
  char * fp = rte_pktmbuf_append(packet, frame_len);
  rte_memcpy(fp, frame, frame_len);

  //  assert(rte_validate_tx_offload(packet)==0);

  /* send packet */
  int send_count = rte_eth_tx_buffer(_device._port_id,
                                     _queue_id,
                                     _tx_buffer,
                                     packet);
  
  /* synchronous flush for the moment */
  send_count += rte_eth_tx_buffer_flush(_device._port_id,
                                        _queue_id,
                                        _tx_buffer);

  if(send_count != 1)
    throw General_exception("rte_eth_tx failure");

  return send_count;
}


int
DPDK::Ethernet_tx_queue::
ethernet_send_raw(struct ether_addr& dst_addr,
                  uint16_t eth_type,
                  struct rte_mbuf *pkt,
                  void * payload,
                  size_t payload_len)
{
  TRACE();
  
  struct ether_hdr *eth_hdr = (struct ether_hdr *)
    rte_pktmbuf_prepend(pkt, (uint16_t)sizeof(struct ether_hdr));

  if(!eth_hdr)
    throw General_exception("ethernet_send: not enough head-room");

  ether_addr_copy(&_device._eth_addr, &eth_hdr->s_addr); // set source MAC address
  ether_addr_copy(&dst_addr, &eth_hdr->d_addr); // set dest MAC address
  eth_hdr->ether_type = rte_be_to_cpu_16(eth_type);

#ifdef TARGET_MELLANOX
  char * payload_offset = rte_pktmbuf_append(pkt, payload_len); /* append two bytes */
  assert(payload_offset);
  rte_memcpy(payload_offset, payload, payload_len);
#else
  struct rte_mbuf * payload_packet = rte_pktmbuf_alloc(_tx_mbuf_pool);
  assert(payload_packet);

  payload_len = 64;
  char * pl = rte_pktmbuf_append(payload_packet, payload_len);

  // /* Chain buffer */  
  if(rte_pktmbuf_chain(pkt, payload_packet)!=0)
    throw General_exception("buffer chaining failed");
#endif

  //  assert(rte_validate_tx_offload(pkt)==0);

  PLOG("formed packet len: %d",pkt->pkt_len);
  
  /* send packet */
  int send_count = rte_eth_tx_buffer(_device._port_id,
                                     _queue_id,
                                     _tx_buffer,
                                     pkt);
  
  /* synchronous flush for the moment */
  send_count += rte_eth_tx_buffer_flush(_device._port_id,
                                        _queue_id,
                                        _tx_buffer);

  if(send_count != 1)
    throw General_exception("rte_eth_tx failure");

  /* release memory */
  //rte_free(fake);

  return send_count;
}


void
DPDK::Ethernet_tx_queue::
bounce_packet(struct rte_mbuf * pkt)
{
  struct ether_hdr *eth_hdr = rte_pktmbuf_mtod(pkt,
                                               struct ether_hdr *);
      
  struct ip_hdr *ip_hdr = rte_pktmbuf_mtod_offset(pkt,
                                                  struct ip_hdr *,
                                                  sizeof(struct ether_hdr));
  
  struct udp_hdr *udp_hdr = rte_pktmbuf_mtod_offset(pkt,
                                                    struct udp_hdr *,
                                                    sizeof(struct ether_hdr)+sizeof(struct ip_hdr));

  /* switch Ethernet addresses */
  char src_addr[64];
  ETHADDR32_COPY(src_addr,  &eth_hdr->s_addr);
  ETHADDR32_COPY(&eth_hdr->s_addr, &eth_hdr->d_addr);
  ETHADDR32_COPY(&eth_hdr->d_addr, src_addr);;


  /* switch IP */
  ip4_addr_p_t tmp;
  IPADDR2_COPY(&tmp, &ip_hdr->dest);  
  IPADDR2_COPY(&ip_hdr->dest, &ip_hdr->src);
  IPADDR2_COPY(&ip_hdr->src, &tmp);
  IPH_CHKSUM_SET(ip_hdr, 0);
  
  /* switch UDP ports */
  //  auto tmp2 = udp_hdr->src_port;
  udp_hdr->dst_port = lwip_htons(100); //udp_hdr->src_port;
  udp_hdr->src_port = udp_hdr->dst_port;
  udp_hdr->dgram_cksum = 0;
  // udp_hdr->dst_port = lwip_htons(dst_port);
  // udp_hdr->dgram_len = lwip_htons(payload_len + 8);   /**< UDP datagram length */

  
  int send_count = rte_eth_tx_buffer(_device._port_id,
                                     _queue_id,
                                     _tx_buffer,
                                     pkt);
  
  /* synchronous flush for the moment */
  send_count += rte_eth_tx_buffer_flush(_device._port_id,
                                        _queue_id,
                                        _tx_buffer);

  assert(send_count == 1);
  //  PLOG("packet bounced OK! (count=%d)", send_count);
}
