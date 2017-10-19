#ifndef __DPDK_RX_H__
#define __DPDK_RX_H__

#include "dpdk_eth.h"

namespace DPDK
{

// forwards decls
//
class Ethernet_device;
class Ethernet_tx_queue;

/** 
 * Ethernet RX queue. This class owns a thread.
 * 
 */
class Ethernet_rx_queue
{
private:
  static constexpr size_t NB_MBUF = 4096;
  static constexpr size_t MEMPOOL_CACHE_SIZE = 256;
  static constexpr uint16_t NB_RXDESC = 128; /* number of receive descriptors per queue */

public:
  /** 
   * Ethernet_rx_queue constructor
   * 
   * @param device Device handle
   * @param queue_id Queue identifier counting from 0
   * @param mempool Memory pool
   * @param tx_queue TX queue used for responses
   */
  Ethernet_rx_queue(Ethernet_device& device,
                    unsigned queue_id,
                    unsigned lcore_id,
		    Ethernet_tx_queue * tx_queue);

  virtual ~Ethernet_rx_queue();

  struct netif * netif() const;

private:
  static int lcore_entry_trampoline(void * arg) {
    return static_cast<Ethernet_rx_queue*>(arg)->lcore_entry(arg);
  }

  int lcore_entry(void * arg);

public:
  static unsigned      _received; // temporary
private:
  Ethernet_device&     _device;
  unsigned             _queue_id;
  unsigned             _lcore_id;
  bool                 _exit_flag;
  struct rte_mempool * _rx_mbuf_pool;
  Ethernet_tx_queue *  _tx_queue;

};

}

#endif
