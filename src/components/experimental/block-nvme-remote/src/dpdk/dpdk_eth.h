#ifndef __DPDK_ETH_H__
#define __DPDK_ETH_H__

#include <stdint.h>
#include <vector>
#include <mutex>
#include <common/exceptions.h>
#include <common/utils.h>
#include <rte_pci.h>
#include <rte_mbuf.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_launch.h>
#include <rte_udp.h>
#include <lwip/netif.h>

#include "dpdk_rx.h"
#include "dpdk_tx.h"

#define IP_TOS             1
#define IP_TTL             2

struct netif;

namespace DPDK
{

typedef struct rte_mbuf packet_t;

#if 0
static rte_mempool * global_big_pool;

static void init_io_pool();

static struct rte_mbuf * alloc_io()
{
  if(!global_big_pool) init_io_pool();
  return rte_pktmbuf_alloc (global_big_pool);
}


static void init_io_pool() 
{
  unsigned num_elements = 8;
  global_big_pool = rte_pktmbuf_pool_create("bigpool",
                                            num_elements, //8192, /* number of element, pow 2 - 1 , 32MB*/
                                            0, /* cache size */
                                            0, //RTE_MBUF_PRIV_ALIGN, /* priv size */
                                            4096, /* data room size */
                                            SOCKET_ID_ANY);
  if(global_big_pool == NULL) {
    PERR("unable to init big pool");
    assert(0);
  }

  //  phys_addr_t last = 0;
  //  	rte_mempool_obj_iter(mp, rte_pktmbuf_init, NULL);
  // for(unsigned i=0;i<num_elements;i++) {
  //   struct rte_mbuf * pbuf = alloc_io();
  //   PLOG("buffer[%u]:%p",i,(void*) rte_mbuf_data_dma_addr(pbuf));
  //   if(last) {
  //     PLOG("delta: %d", rte_mbuf_data_dma_addr(pbuf) - last);
  //   }
  //   last = rte_mbuf_data_dma_addr(pbuf);
  //   rte_pktmbuf_free(pbuf);
  // }
      
}


static void free_io(struct rte_mbuf * buf)
{
  //  rte_pktmbuf_free(struct rte_mbuf *m)
}
#endif


/** 
 * Ethernet device instance
 * 
 */
class Ethernet_device
{
  friend class Ethernet_rx_queue;
  friend class Ethernet_tx_queue;
  
private:
  static constexpr bool option_DEBUG = true;

public:

  /** 
   * Constructor
   * 
   */
  Ethernet_device(const char * pci_addr,
                  unsigned num_rx_queues,
                  unsigned num_tx_queues,
                  rte_cpuset_t rx_thread_core_mask);

  /** 
   * Destructor will clean up EAL resources
   * 
   */
  virtual ~Ethernet_device();

  /** 
   * Get hold of TX queue queue
   * 
   * @param queue_id Queue identifier, counting from 0.
   * 
   * @return Queue instance.  Normal dtor release.
   */
  Ethernet_tx_queue * get_tx_queue(unsigned queue_id);


  /** 
   * Get hold of RX queue
   * 
   * @param queue_id 
   * 
   * @return 
   */
  Ethernet_rx_queue * get_rx_queue(unsigned queue_id);

  
  /** 
   * Get number of TX queues 
   * 
   * @return Number of queues
   */
  size_t num_tx_queues() {  return _tx_queues.size();  }

  /** 
   * Get number of RX queues 
   * 
   * @return Number of queues
   */
  size_t num_rx_queues()  {  return _rx_queues.size();  }

  /** 
   * Access MAC address
   * 
   * 
   * @return MAC address
   */
  struct ether_addr mac_addr() { return _eth_addr; }
  
  void test_tx2(void * payload, size_t payload_len);
  void test_rx();

  
  void test_netcore();
  void test_netcore_rx();

  unsigned udp_send(unsigned queue, const ip4_addr_t * dst_ip,
                    uint16_t src_port,
                    uint16_t dst_port, void * payload, size_t payload_len);

  struct udp_hdr * udp_prepare(const ip4_addr_t * dst_ip,
                               uint16_t src_port,
                               uint16_t dst_port,
                               struct rte_mbuf * packet,
                               size_t payload_len);
  
  /** 
   * Get the port id for the device
   * 
   * 
   * @return 
   */
  uint8_t port_id() const { return _port_id; }
  
private:
  void initialize_port(unsigned num_rx_queues, unsigned num_tx_queues, rte_cpuset_t rx_thread_core_mask);
  void netif_init();

  
protected:
  uint8_t                        _port_id;
  
  struct rte_eth_dev_info        _dev_info;
  struct rte_eth_conf            _port_conf;
  struct rte_pci_addr            _dev_addr;
  struct ether_addr              _eth_addr; 

  std::vector<Ethernet_tx_queue *> _tx_queues;
  std::vector<Ethernet_rx_queue *> _rx_queues;

  struct netif *_netif; // LWIP
  char pad[255]; /* bug hack */
  
  std::mutex _netif_lock;

  inline void LWIP_LOCK() { _netif_lock.lock(); }
  inline void LWIP_UNLOCK() { _netif_lock.unlock(); }
};



}

#endif // __DPDK_ETH_H__
