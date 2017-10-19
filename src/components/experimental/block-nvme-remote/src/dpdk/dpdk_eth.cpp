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
#include <rte_mbuf.h>

#include "nvme_buffer.h"
#include "dpdk_eth.h"
#include "eal_init.h"
#include "csem.h"

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
#include "dpdk_rx.h"

namespace LWIP
{
#include <lwip/udp.h>
}

using namespace DPDK;

static void ip4_print(struct ip_hdr *iphdr);
static void eth_print(struct ether_hdr * ethhdr);
static void eth_addr_print(struct eth_addr * ethaddr);

static void
tcp_timeout(void *data)
{
  LWIP_UNUSED_ARG(data);
#if TCP_DEBUG && LWIP_TCP
  tcp_debug_print_pcbs();
#endif /* TCP_DEBUG */
  sys_timeout(500, tcp_timeout, NULL);
}


DPDK::Ethernet_device::
Ethernet_device(const char * pci_addr,
                unsigned num_rx_queues,
                unsigned num_tx_queues,
                rte_cpuset_t rx_thread_core_mask)
{
  int rc;

  DPDK::eal_init(0); /* initialize DPDK memory and drivers */

  assert(CPU_COUNT(&rx_thread_core_mask) == (signed) num_rx_queues);

  /* check that the creating thread is on the master core */
  if(rte_get_master_lcore() != rte_lcore_id()) {
    throw Constructor_exception("constructor must be called from DPDK master lcore only.");
  }
  
  uint8_t nb_ports = rte_eth_dev_count();
  PLOG("Compatible Ethernet device count: %i", nb_ports);

  if(nb_ports == 0)
    throw Constructor_exception("no compatible devices.");

  rc = eal_parse_pci_BDF(pci_addr, &_dev_addr);
  if(rc)
    throw Constructor_exception("bad address");


  bool found = false;
  for(unsigned p=0; p < nb_ports; p++) {

    rte_eth_dev_info_get(p, &_dev_info);
    
    PLOG("Compatible Ethernet device: %02x:%02x.%x",
         _dev_info.pci_dev->addr.bus,
         _dev_info.pci_dev->addr.devid,
         _dev_info.pci_dev->addr.function);

    PLOG("\tTX IP checksum offload=%s",
         _dev_info.tx_offload_capa & DEV_TX_OFFLOAD_IPV4_CKSUM ? "yes" : "no");

    PLOG("\tTX UDP checksum offload=%s",
         _dev_info.tx_offload_capa & DEV_TX_OFFLOAD_UDP_CKSUM ? "yes" : "no");
    
    if(rte_eal_compare_pci_addr(&_dev_addr, &_dev_info.pci_dev->addr)==0) {
      found = true;
      _port_id = p;
      break;
    }
  }

  if(!found)
    throw Constructor_exception("device not found.");
    
  PLOG("Attached to Ethernet device: %02x:%02x.%x",
       _dev_info.pci_dev->addr.bus,
       _dev_info.pci_dev->addr.devid,
       _dev_info.pci_dev->addr.function);
  PLOG("Max RX queues:%u", _dev_info.max_rx_queues);
  PLOG("Max TX queues:%u", _dev_info.max_tx_queues);

  if(num_rx_queues > _dev_info.max_rx_queues ||
     num_tx_queues > _dev_info.max_tx_queues)
     throw Constructor_exception("exceeded number of queues supported by device");

  /* initialize queues etc. */
  initialize_port(num_rx_queues,num_tx_queues,rx_thread_core_mask);
  
  /* check link status */
  struct rte_eth_link link;
  __builtin_bzero(&link,sizeof(link));
  rte_eth_link_get(_port_id, &link);
  if (link.link_status) {
    PLOG("Port %d Link Up - speed %u Mbps - %s", (uint8_t)_port_id,
         (unsigned)link.link_speed,
         (link.link_duplex == ETH_LINK_FULL_DUPLEX) ?
         ("full-duplex") : ("half-duplex\n"));
  }
  else {
    PLOG("Port %d Link Down", (uint8_t)_port_id);
  }

  /* configure flow control */
  struct rte_eth_fc_conf fc_conf;
  rc = rte_eth_dev_flow_ctrl_get(_port_id, &fc_conf);
  if(rc)
    throw General_exception("rte_eth_dev_flow_ctrl_get: failed");

  #if 0
  fc_conf.mode = RTE_FC_FULL;
  fc_conf.high_water = 80*510 / 100;
  fc_conf.low_water = 60*510 / 100;
  fc_conf.pause_time = 1337;
  fc_conf.send_xon = 0;
  #endif
  rc = rte_eth_dev_flow_ctrl_set(_port_id,&fc_conf);
  if(rc)
    throw General_exception("rte_eth_dev_flow_ctrl_set: failed");

  PLOG("flow fc params: hw=%d lw=%d pt=%d send_xon=%d mac_ctrl_frame_fwd=%d autoneg=%d, mode=%d",
       fc_conf.high_water, fc_conf.low_water, fc_conf.pause_time, fc_conf.send_xon, fc_conf.mac_ctrl_frame_fwd,
       fc_conf.autoneg, fc_conf.mode);
       
  
  /* initialize netif */
  netif_init();

  //test_rx();
  //PLOG("test RX done.");
}

DPDK::Ethernet_device::~Ethernet_device()
{
  rte_eth_dev_stop(_port_id);
  PLOG("Device stopped OK.");
  
  /* clean up queues */
  for(auto& q: _tx_queues)
    delete q;

  for(auto& q: _rx_queues)
    delete q;

  PLOG("Queue cleanup OK.");
}


struct port_statistics {
  uint64_t tx;
  uint64_t rx;
  uint64_t dropped;
} __rte_cache_aligned;

static struct port_statistics port_statistics;



void DPDK::Ethernet_device::initialize_port(unsigned num_rx_queues,
                                            unsigned num_tx_queues,
                                            rte_cpuset_t rx_thread_core_mask)
{
  int rc;
  
  PLOG("Initializing port..");

  /* configure port */
  __builtin_bzero(&_port_conf,sizeof(struct rte_eth_conf));

  //  _port_conf.link_speeds =  rte_eth_speed_bitflag(100000, ETH_LINK_FULL_DUPLEX);
  
  _port_conf.rxmode.split_hdr_size = 0;
  _port_conf.rxmode.header_split   = 0; /**< Header Split disabled */
  _port_conf.rxmode.hw_ip_checksum = 0; /**< IP checksum offload disabled */
  _port_conf.rxmode.hw_vlan_filter = 0; /**< VLAN filtering disabled */
  _port_conf.rxmode.hw_vlan_strip = 0; /**< VLAN strip  */
  _port_conf.rxmode.jumbo_frame    = 0; /**< Jumbo Frame Support disabled */
  _port_conf.rxmode.hw_strip_crc   = 0; /**< CRC stripped by hardware */

  _port_conf.rx_adv_conf.rss_conf.rss_key = NULL;
  _port_conf.rx_adv_conf.rss_conf.rss_key_len = 0;
  _port_conf.rx_adv_conf.rss_conf.rss_hf = ETH_RSS_IP;

  // checkum offloading is set by default
  _port_conf.txmode.mq_mode = ETH_MQ_TX_NONE;
  //  _port_conf.intr_conf.lsc = 1; /**< link status interrupt feature enabled */

  rc = rte_eth_dev_configure(_port_id,
                             num_rx_queues,
                             num_tx_queues,
                             &_port_conf);
  if(rc)
    throw General_exception("rte_eth_dev_configure failed");

  PLOG("Device configured OK.");

  rte_eth_macaddr_get(_port_id,&_eth_addr);
  PLOG("MAC address: %02X:%02X:%02X:%02X:%02X:%02X",
       _eth_addr.addr_bytes[0],
       _eth_addr.addr_bytes[1],
       _eth_addr.addr_bytes[2],
       _eth_addr.addr_bytes[3],
       _eth_addr.addr_bytes[4],
       _eth_addr.addr_bytes[5]);

  /* create TX queues */
  for(unsigned i=0;i<num_tx_queues;i++) {
    _tx_queues.push_back(new Ethernet_tx_queue(*this, i));
  }

  /* create RX queues */
  unsigned core = 0;
  for(unsigned i=0;i<num_rx_queues;i++) {
    while(!CPU_ISSET(core, &rx_thread_core_mask)) core++;
    
    _rx_queues.push_back(new Ethernet_rx_queue(*this, i, core, _tx_queues[i]));
    core++;
  }
  

  /* start device */
  rc = rte_eth_dev_start(_port_id);
  if (rc < 0)
    throw General_exception("unable to start device");

  rte_eth_promiscuous_enable(_port_id);
}

static inline void
ether_string_to_addr(const char *straddr, 
                     struct ether_addr *eth_addr)
{
	sscanf(straddr, "%02X:%02X:%02X:%02X:%02X:%02X",

         (unsigned int*) &eth_addr->addr_bytes[1],
         (unsigned int*) &eth_addr->addr_bytes[2],
         (unsigned int*) &eth_addr->addr_bytes[3],
         (unsigned int*) &eth_addr->addr_bytes[4],
         (unsigned int*) &eth_addr->addr_bytes[5]);
}

DPDK::Ethernet_tx_queue *
DPDK::Ethernet_device::get_tx_queue(unsigned queue_id)
{
  if(queue_id > _tx_queues.size())
    throw General_exception("invalid queue id");
  
  return _tx_queues[queue_id];
}


DPDK::Ethernet_rx_queue *
DPDK::Ethernet_device::get_rx_queue(unsigned queue_id)
{
  if(queue_id > _tx_queues.size())
    throw General_exception("invalid queue id");
  
  return _rx_queues[queue_id];
}


static err_t netif_output(struct netif *netif, struct pbuf *p)
{
  TRACE();
  LINK_STATS_INC(link.xmit);

  // use queue 0 for stack generated traffic, e.g., ARP, ICMP
  DPDK::Ethernet_tx_queue * txqueue = static_cast<DPDK::Ethernet_tx_queue *>(netif->arg);
  assert(txqueue);
  txqueue->ethernet_send_frame(p->payload,p->len);
  
  return ERR_OK;
}

static err_t
default_netif_init(struct netif *netif)
{
  assert(netif);
  netif->linkoutput = netif_output;
  netif->output = etharp_output;
  netif->mtu = 1500;
  netif->flags = NETIF_FLAG_BROADCAST | NETIF_FLAG_ETHARP | NETIF_FLAG_LINK_UP;
  netif->hwaddr_len = ETHARP_HWADDR_LEN;
  netif->name[0] = 'e';
  netif->name[1] = 'n';

  return ERR_OK;
}


void
DPDK::Ethernet_device::
netif_init()
{
  /* initialize LWIP stack */
  lwip_init(); 

  _netif = new struct netif;
  _netif->arg = (void *) get_tx_queue(0);
  
  /* configure MTU */
  uint16_t mtu = 0;
  rte_eth_dev_get_mtu(_port_id, &mtu);
  assert(mtu > 0);
  _netif->mtu = mtu;

  /* configure MAC */
  SMEMCPY(_netif->hwaddr, &_eth_addr, sizeof(_netif->hwaddr));
  _netif->hwaddr_len = sizeof(_netif->hwaddr);

  /* initialize TCP stack */
  #if !defined(NO_SYS)
  {
    Semaphore s;
    tcpip_init([](void*arg){ ((Semaphore*)arg)->post(); },(void*)&s);
    s.wait();
    PLOG("LWIP tcpip initialized OK!");
  }
  #endif

  /* TODO here: configure from a file */
  /* configure IPv4 */
  static ip4_addr_t ipaddr, netmask, gw;
  IP4_ADDR(&gw, 11,0,0,1);
  IP4_ADDR(&ipaddr, 11,0,0,2);
  IP4_ADDR(&netmask, 255,0,0,0);


  netif_set_default(netif_add(_netif, &ipaddr, &netmask,
                              &gw, NULL, default_netif_init,
#if !defined(NO_SYS)
                              tcpip_input
#else
                              NULL
#endif
                              ));
  netif_set_up(_netif);

  sys_timeout(500, tcp_timeout, NULL);
}

void
DPDK::Ethernet_device::
test_rx()
{
  const unsigned BURST_SIZE = 5;
  
  while(1) {

    //    PLOG("Waiting to rx packet..");
    struct rte_mbuf * p_rx_buffers[BURST_SIZE];
    uint16_t rc = rte_eth_rx_burst(_port_id,
                                   0, //uint16_t  	queue_id,
                                   p_rx_buffers,
                                   BURST_SIZE);
    
    //    PLOG("rte_eth_rx_burst return code:%u",rc);

    
    for(unsigned p=0;p<rc;p++) {

      struct ether_hdr *eth_hdr = rte_pktmbuf_mtod(p_rx_buffers[p],
                                                   struct ether_hdr *);

      char src_addr[64];
      ether_format_addr(src_addr, 64, &eth_hdr->s_addr);

      char dst_addr[64];
      ether_format_addr(dst_addr, 64, &eth_hdr->d_addr);

      unsigned char * payload = ((unsigned char*)p_rx_buffers[p]) + sizeof(struct ether_hdr);
      PINF("Rx packet: %s - %s [%.4x]: %.2x %.2x %.2x %.2x %.2x %.2x",
           dst_addr, src_addr, rte_cpu_to_be_16(eth_hdr->ether_type),
           payload[0],payload[1],payload[2],payload[3],payload[4],payload[5]);
    }
    
    /* Free buffers. */
    for(unsigned p=0;p<rc;p++) {
      rte_pktmbuf_free(p_rx_buffers[p]);
    }
  }
}

#define BOND_IP_1       10
#define BOND_IP_2       0
#define BOND_IP_3       0
#define BOND_IP_4       3

void
DPDK::Ethernet_device::
test_tx2(void *payload, size_t payload_len)
{
  TRACE();

  auto q = get_tx_queue(0);

  struct ether_addr dst_addr;
  //  ether_string_to_addr("24:8a:07:91:65:04", &dst_addr); // ribbit1
  //ether_string_to_addr("e4:1f:13:b4:a5:e8", &dst_addr); // compute4
  ether_string_to_addr("ff:ff:ff:ff:ff:ff", &dst_addr); 


  int COUNT = 20;
  
  //#define USE_BURST  
#ifdef USE_BURST

  packet_t * pkts[COUNT];
  void * payloads[COUNT];
  size_t payload_lens[COUNT];
  
  for(auto i=0;i<COUNT;i++) {
    pkts[i] = q->allocate_packet();
    payloads[i] = payload;
    payload_lens[i] = payload_len;
  }

  q->ethernet_send_burst_raw(dst_addr,
                             0xAABB,
                             COUNT,
                             (packet_t **)pkts,
                             (void**)payloads,
                             (size_t *)payload_lens);
  
#else
  for(int i=0;i<COUNT;i++) {
    struct rte_mbuf * pkt = q->allocate_packet();
    //    *((cpu_time_t *)payload) = rdtsc();
    q->ethernet_send_raw(dst_addr, 0xAABB, pkt, payload, payload_len);
    q->release_packet(pkt);
  }
#endif
}




void
DPDK::Ethernet_device::
test_netcore()
{
  const auto PAYLOAD_SIZE = 64;
  void * buffer = Nvme_buffer::allocate_io_buffer(PAYLOAD_SIZE);
  memset(buffer,0xDD, PAYLOAD_SIZE);
  //  test_tx2(buffer, PAYLOAD_SIZE);

  static ip4_addr_t dst_ip;
  IP4_ADDR(&dst_ip, 11,0,0,2);

  udp_send(0,&dst_ip, 999, 10000, buffer, PAYLOAD_SIZE);

  sleep(10);
}

void
DPDK::Ethernet_device::
test_netcore_rx()
{
  sleep(100);
}


struct udp_hdr *
DPDK::Ethernet_device::
udp_prepare(const ip4_addr_t * dst_ip,
            uint16_t src_port,
            uint16_t dst_port,
            struct rte_mbuf * packet,
            size_t payload_len)
{
  ip4_addr_t * resolved_ip;
  struct netif* resolved_netif;
  struct eth_addr* resolved_ethaddr;

  /* lookup destination mac through ARP */
  s8_t valid, idx = 127;
  unsigned attempts = 0;
  do {
    if(idx != 127) {
      usleep(100000);
      LWIP_LOCK();
      sys_check_timeouts();
      LWIP_UNLOCK();
    }
    
    idx = etharp_query(_netif, dst_ip, NULL);
    assert(idx >= 0);
    LWIP_LOCK();
    valid = etharp_get_entry(idx,&resolved_ip,&resolved_netif,&resolved_ethaddr);
    LWIP_UNLOCK();
    if(!valid)
      PLOG("No ARP response.");
    attempts++;
    if(attempts> 3) sleep(1);
    if(attempts > 6) break;
  } while(valid != 1);

  if(!valid) {
    PERR("udp_prepare failed: no ARP response");
    return NULL;
  }

  eth_addr_print(resolved_ethaddr);

  /* create Ethernet header */
  struct ether_hdr *eth_hdr = (struct ether_hdr *)
    rte_pktmbuf_append(packet, sizeof(struct ether_hdr));

  if(!eth_hdr)
    throw General_exception("ethernet_send: not enough head-room");

  ether_addr_copy((const struct ether_addr *)&resolved_netif->hwaddr,
                  &eth_hdr->s_addr); // set source MAC address
  
  ether_addr_copy((const struct ether_addr *)resolved_ethaddr,
                  &eth_hdr->d_addr); // set dest MAC address
  eth_hdr->ether_type = rte_be_to_cpu_16(ETHTYPE_IP);

  /* create IP header */
  struct ip_hdr *ip_hdr = (struct ip_hdr *)
    rte_pktmbuf_append(packet, sizeof(struct ip_hdr));

  IPH_TTL_SET(ip_hdr, UDP_TTL);
  IPH_PROTO_SET(ip_hdr, IP_PROTO_UDP);
  IPH_VHL_SET(ip_hdr, 4, IP_HLEN / 4);
  IPH_TOS_SET(ip_hdr, IP_TOS); // or IPTOS_THROUGHPUT ?
  ip4_addr_copy(ip_hdr->dest, *dst_ip);
  ip4_addr_copy(ip_hdr->src, resolved_netif->ip_addr);
  IPH_LEN_SET(ip_hdr, lwip_htons(IP_HLEN + sizeof(struct udp_hdr) + payload_len));

  ip4_print(ip_hdr);
  
  /* create UDP header */
  struct udp_hdr *udp_hdr = (struct udp_hdr *)
    rte_pktmbuf_append(packet, sizeof(struct udp_hdr));

  udp_hdr->src_port = lwip_htons(src_port);
  udp_hdr->dst_port = lwip_htons(dst_port);
  udp_hdr->dgram_len = lwip_htons(payload_len + 8);   /**< UDP datagram length */

  /* in UDP, 0 checksum means 'no checksum' */
  udp_hdr->dgram_cksum = 0x0000;

  /* set up lengths for HW checksumming */
  packet->l3_len = sizeof(struct ip_hdr); //IP_HLEN;
  packet->l2_len = sizeof(struct ether_hdr);

  return udp_hdr;
}

unsigned
DPDK::Ethernet_device::
udp_send(unsigned queue,
         const ip4_addr_t * dst_ip,
         uint16_t src_port,
         uint16_t dst_port,
         void * payload,
         size_t payload_len)
{
  TRACE();
  PLOG("udp_send: port=%u payload_len=%ld",dst_port,payload_len);
  
  Ethernet_tx_queue * txq = _tx_queues[queue];
  assert(txq);

  struct rte_mbuf * rte_packet = txq->allocate_packet();
  
  /* use ARP to resolve mac address */
  struct udp_hdr * udp_hdr = udp_prepare(dst_ip, src_port, dst_port, rte_packet, payload_len);

  /* set flags for HW checksumming */
  rte_packet->ol_flags |= PKT_TX_IPV4 | PKT_TX_IP_CKSUM | PKT_TX_UDP_CKSUM;

  //#define USE_BURST
#ifdef USE_BURST
  PLOG("sending burst");
  std::vector<Ethernet_tx_queue::payload_t> payload_vector;
  for(unsigned i=0;i<128;i++)
    payload_vector.push_back({payload,payload_len});

  int sent = 0;
  for(unsigned i=0;i<1;i++) {
    sent += txq->ethernet_send_burst_ex(rte_packet, payload_vector);
    PLOG("burst sent: %d",sent);
  }
  
  return sent;  
#else
  // send clones
  rte_mbuf* clones[100];
  for(unsigned i=0;i<100;i++) {
    clones[i] = txq->clone_packet(rte_packet);
  }

  for(unsigned i=0;i<10;i++) {
    Ethernet_rx_queue::_received=0;
    *((uint64_t*)payload) = rdtsc();
    mb();
    txq->ethernet_send_frame_ex(clones[i], payload, payload_len);
    while(Ethernet_rx_queue::_received == 0) cpu_relax();
  }
  *((uint64_t*)payload) = rdtsc();
  return txq->ethernet_send_frame_ex(rte_packet, payload, payload_len);
#endif
}

static void eth_print(struct ether_hdr * ethhdr)
{
PINF("Ethernet header:");
PINF("+---------------------------------------+");
PINF("| %02X:%02X:%02X:%02X:%02X:%02X | %02X:%02X:%02X:%02X:%02X:%02X | (dst addr, src addr)",
                         ethhdr->d_addr.addr_bytes[0],
                         ethhdr->d_addr.addr_bytes[1],
                         ethhdr->d_addr.addr_bytes[2],
                         ethhdr->d_addr.addr_bytes[3],
                         ethhdr->d_addr.addr_bytes[4],
                         ethhdr->d_addr.addr_bytes[5],
                         ethhdr->s_addr.addr_bytes[0],
                         ethhdr->s_addr.addr_bytes[1],
                         ethhdr->s_addr.addr_bytes[2],
                         ethhdr->s_addr.addr_bytes[3],
                         ethhdr->s_addr.addr_bytes[4],
                         ethhdr->s_addr.addr_bytes[5]
                         );
PINF("+---------------------------------------+");
PINF("| %04X |                                | (type)",lwip_ntohs(ethhdr->ether_type));
PINF("+---------------------------------------+");
}

static void eth_addr_print(struct eth_addr * ethaddr)
{
  PINF("Ethernet addr:");
  PINF("+---------------------------+");
  PINF("| %02X:%02X:%02X:%02X:%02X:%02X |",
       ethaddr->addr[0],
       ethaddr->addr[1],
       ethaddr->addr[2],
       ethaddr->addr[3],
       ethaddr->addr[4],
       ethaddr->addr[5]
       );
  PINF("+---------------------------+");
}


static void ip4_print(struct ip_hdr *iphdr)
{
PINF("IP header:");
PINF("+-------------------------------+");
PINF("|%2" S16_F " |%2" S16_F " |  0x%02" X16_F " |     %5" U16_F "     | (v, hl, tos, len)",
                    (u16_t)IPH_V(iphdr),
                    (u16_t)IPH_HL(iphdr),
                    (u16_t)IPH_TOS(iphdr),
                    lwip_ntohs(IPH_LEN(iphdr)));
PINF("+-------------------------------+");
PINF("|    %5" U16_F "      |%" U16_F "%" U16_F "%" U16_F "|    %4" U16_F "   | (id, flags, offset)",
                    lwip_ntohs(IPH_ID(iphdr)),
                    (u16_t)(lwip_ntohs(IPH_OFFSET(iphdr)) >> 15 & 1),
                    (u16_t)(lwip_ntohs(IPH_OFFSET(iphdr)) >> 14 & 1),
                    (u16_t)(lwip_ntohs(IPH_OFFSET(iphdr)) >> 13 & 1),
                    (u16_t)(lwip_ntohs(IPH_OFFSET(iphdr)) & IP_OFFMASK));
PINF("+-------------------------------+");
PINF("|  %3" U16_F "  |  %3" U16_F "  |    0x%04" X16_F "     | (ttl, proto, chksum)",
                    (u16_t)IPH_TTL(iphdr),
                    (u16_t)IPH_PROTO(iphdr),
                    lwip_ntohs(IPH_CHKSUM(iphdr)));
PINF("+-------------------------------+");
PINF("|  %3" U16_F "  |  %3" U16_F "  |  %3" U16_F "  |  %3" U16_F "  | (src)",
                    ip4_addr1_16(&iphdr->src),
                    ip4_addr2_16(&iphdr->src),
                    ip4_addr3_16(&iphdr->src),
                    ip4_addr4_16(&iphdr->src));
PINF("+-------------------------------+");
PINF("|  %3" U16_F "  |  %3" U16_F "  |  %3" U16_F "  |  %3" U16_F "  | (dest)",
                    ip4_addr1_16(&iphdr->dest),
                    ip4_addr2_16(&iphdr->dest),
                    ip4_addr3_16(&iphdr->dest),
                    ip4_addr4_16(&iphdr->dest));
PINF("+-------------------------------+");
}



// SCRAPS

#if 0
/** 
 * Transmit a packet on queue 0
 * 
 */
void
DPDK::Ethernet_device::
test_tx()
{
  PLOG("Performing test_tx ....");
  unsigned queue_id = 4;

  auto queue = get_tx_queue(queue_id);

  assert(_tx_buffer[queue_id]);
  
  for(unsigned i=0;i<20;i++) {
    struct rte_mbuf *tx_pkt = queue->allocate_packet();
    
    struct ether_hdr *eth_hdr = rte_pktmbuf_mtod(tx_pkt, struct ether_hdr *);
    struct ether_addr dst_addr;
    //ether_string_to_addr("08:94:ef:32:49:97", &dst_addr);
    ether_string_to_addr("ff:ff:ff:ff:ff:ff", &dst_addr);
    //ether_string_to_addr("24:8A:07:91:65:04", &dst_addr); // compute4
  
    /* form an Ethernet header */
    ether_addr_copy(&_eth_addr, &eth_hdr->s_addr); // set source MAC address
    ether_addr_copy(&dst_addr, &eth_hdr->d_addr); // set dest MAC address
    eth_hdr->ether_type = rte_cpu_to_be_16(ETHER_TYPE_ARP);


    struct arp_hdr *arp_hdr = rte_pktmbuf_mtod_offset(tx_pkt,
                                                      struct arp_hdr *,
                                                      sizeof(struct ether_hdr));
    memset(arp_hdr, 0, sizeof(struct arp_hdr));
    arp_hdr->arp_hrd = rte_cpu_to_be_16(ARP_HRD_ETHER);
    arp_hdr->arp_pro = rte_cpu_to_be_16(0x800);    /* format of protocol address */
    arp_hdr->arp_hln = 6;    /* length of hardware address */
	  arp_hdr->arp_pln = 4;    /* length of protocol address */
    arp_hdr->arp_op = rte_cpu_to_be_16(ARP_OP_REQUEST);
    
    uint32_t bond_ip = BOND_IP_1 | (BOND_IP_2 << 8) |
      (BOND_IP_3 << 16) | (BOND_IP_4 << 24);

    uint32_t bond_ip2 = BOND_IP_1 | (BOND_IP_2 << 8) |
      (BOND_IP_3 << 16) | (99 << 24);
        
    
    ether_addr_copy(&_eth_addr, &arp_hdr->arp_data.arp_sha);
    arp_hdr->arp_data.arp_sip = bond_ip;
    arp_hdr->arp_data.arp_tip = bond_ip2;

    tx_pkt->pkt_len = sizeof(struct ether_hdr) + sizeof(struct arp_hdr);
    tx_pkt->data_len = tx_pkt->pkt_len;

    tx_pkt->nb_segs = 1;

    __rte_mbuf_sanity_check(tx_pkt, 1);

    assert(0);
    auto q = get_queue(0);
    q->ethernet_send_raw(dst_addr, 0xAABB, pkt, payload, payload_len);
    // rc = rte_eth_tx_buffer(_port_id,
    //                        queue_id,
    //                        _tx_buffer[queue_id],
    //                        tx_pkt);
    //    PLOG("tx buffer rc=%u",rc);
  }

  // rc = rte_eth_tx_buffer_flush(_port_id,
  //                              queue_id,
  //                              _tx_buffer[queue_id]);
  //  PLOG("flush rc=%u",rc);
  
}

#endif
