#ifndef __DPDK_TX_H__
#define __DPDK_TX_H__

#include "dpdk_eth.h"

namespace DPDK
{

class Ethernet_device;

/** 
 * Ethernet TX queue
 * 
 */
class Ethernet_tx_queue
{
  static constexpr size_t   NB_MBUF = (2*1024);
  static constexpr size_t   MEMPOOL_CACHE_SIZE = 256;
  static constexpr size_t   MAX_PKT_BURST = 256;
  static constexpr size_t   BURST_TX_DRAIN_US = 100; /* TX drain every ~100us */
  static constexpr uint16_t NB_TXDESC = 256;
  
public:

  /** 
   * Ethernet_tx_queue constructor
   * 
   * @param device Device handle
   * @param queue_id Queue identifier counting from 0
   * @param mempool Memory pool 
   * @param tx_buffer Transmit buffer descriptor
   */
  Ethernet_tx_queue(Ethernet_device& device,
                    unsigned queue_id);

  /** 
   * Destructor
   * 
   * 
   * @return 
   */
  virtual ~Ethernet_tx_queue();
  
  /** 
   * Allocate an IO packet
   * 
   * 
   * @return Pointer to IO packet (struct rte_mbuf)
   */
  struct rte_mbuf * allocate_packet();


  /** 
   * Clone a packet
   * 
   * @param mb Existing mbuf
   * 
   * @return New mbuf
   */
  struct rte_mbuf * clone_packet(struct rte_mbuf *mb);

  /** 
   * Release IO packet
   * 
   * @param p Pointer to IO packet
   */
  void release_packet(struct rte_mbuf * p);

  /** 
   * Encapsulate payload in ethernet frame and send
   * 
   * @param dst_addr Destination MAC address
   * @param eth_type Ethernet frame type
   * @param p Packet
   * @param dst_addr Destination MAC address 
   * @param eth_type Ethernet frame type
   * @param payload Payload (e.g., IO buffer)
   * @param payload_len Payload length in bytes
   * 
   * @return Number of packets sent, < 0 on error
   */
  int ethernet_send_raw(struct ether_addr& dst_addr,
                        uint16_t eth_type,
                        struct rte_mbuf * pkt,
                        void * payload,
                        size_t payload_len);

  int ethernet_send_frame(void * frame,
                          size_t frame_len);

  /** 
   * Send a frame with a pre-built header segment
   * 
   * @param header_segment Header segment
   * @param frame Payload to append
   * @param frame_len Length of payload
   * 
   * @return Number of packets sent
   */
  int ethernet_send_frame_ex(struct rte_mbuf* header_segment,
                             void * frame,
                             size_t frame_len);

  typedef struct {
    void * data;
    size_t len;
  } payload_t;
  
  int ethernet_send_burst_ex(struct rte_mbuf* header_segment,
                             std::vector<payload_t>& payload_vector);

  /** 
   * Transmit a burst of frames
   * 
   * @param dst_addr Destination MAC address
   * @param eth_type Ethernet frame type
   * @param burst_size Size of burst in packets
   * @param pkt Array of packets
   * @param payload Array of payloads
   * @param payload_len Array of payload lengths
   * 
   * @return Number of packets sent, < 0 on error
   */
  int ethernet_send_burst_raw(struct ether_addr& dst_addr,
                              uint16_t eth_type,
                              size_t burst_size,
                              struct rte_mbuf ** pkt,
                              void ** payload,
                              size_t * payload_len);


  void bounce_packet(struct rte_mbuf* ingress_packet);
  
private:
  Ethernet_device&               _device;
  unsigned                       _queue_id;
  struct rte_mempool *           _tx_mbuf_pool;
  struct rte_eth_dev_tx_buffer * _tx_buffer;  
  struct rte_eth_txconf          _tx_conf;
};

}
  
#endif
