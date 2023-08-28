#include <pthread.h>
#include <stdlib.h>

#include <rte_malloc.h>
#include <rte_sft.h>
#include <rte_net.h>
#include <rte_ether.h>
#include <rte_mbuf.h>
#include <rte_memcpy.h>
#include <rte_ethdev.h>
#include <netinet/tcp.h> // For TCP header

#include <doca_buf_inventory.h>
#include <doca_log.h>
#include <doca_mmap.h>
#include <doca_dpdk.h>

#include "dpi_worker.h"
#include "dpdk_utils.h"
#include "offload_rules.h"
#include "doca_compress.h"
#include "process_data.h" 
#include "process_buffer.h" 

DOCA_LOG_REGISTER(DWRKR);

#define BURST_SIZE 1024			/* Receive burst size */
#define NETFLOW_UPDATE_RATE 2000	/* Send netflow update every 2000 packet */
#define RECEIVER (0x0)			/* Receiver index */
#define INITIATOR (0x1)			/* Initiator index */
#define SFT_ZONE 0xCAFE			/* zone for sft (arbitrary value) */
#define SFT_FLOW_INFO_CLIENT_ID 0xF	/* Flow info client id (arbitrary value) */
#define MEMPOOL_NAME_MAX_LEN 100	/* The mempool name maximum byte length */
#define UDP_HEADER_SIZE (8)			/* UDP header size = 8 bytes (64 bits) */
#define USER_PCI_ADDR_LEN 7			/* User PCI address string length */
#define PCI_ADDR_LEN (USER_PCI_ADDR_LEN + 1)
#define PKT_TX_TCP_CKSUM     (1ULL << 52) 
#define PKT_TX_SCTP_CKSUM    (2ULL << 52) 
#define PKT_TX_UDP_CKSUM     (3ULL << 52) 
#define PKT_TX_L4_MASK       (3ULL << 52) 
#define PKT_TX_IP_CKSUM      (1ULL << 54)
 
#define PKT_TX_IPV4          (1ULL << 55)
 
#define PKT_TX_IPV6          (1ULL << 56)
 
#define PKT_TX_VLAN_PKT      (1ULL << 57) 
#define PKT_TX_OUTER_IP_CKSUM   (1ULL << 58)
 
#define PKT_TX_OUTER_IPV4   (1ULL << 59) 
#define PKT_TX_OUTER_IPV6    (1ULL << 60)

/* Buffers packet for future transmission */
#define TX_BUFFER_PKT(m, ctx) rte_eth_tx_buffer(m->port ^ 1, ctx->queue_id, ctx->tx_buffer[m->port ^ 1], m)

#define DPI_WORKER_ALWAYS_INLINE inline __attribute__((always_inline))

// Macro to create a 32-bit IPv4 address in network byte order
#define MAKE_IPV4_ADDRESS(a, b, c, d) (((a) << 24) | ((b) << 16) | ((c) << 8) | (d))

#define TCP_FIN_FLAG 0x01
#define TCP_SYN_FLAG 0x02
#define TCP_RST_FLAG 0x04
#define TCP_PSH_FLAG 0x08
#define TCP_ACK_FLAG 0x10
#define TCP_URG_FLAG 0x20


// Initial size of the dynamic buffer
#define INITIAL_BUFFER_SIZE 1024
#define PACKET_SIZE 1500 // Define your desired packet size here

unsigned char *data_buffer;
size_t total_data_size;
size_t buffer_size;
bool ackcaught;

pthread_mutex_t log_lock;
volatile bool force_quit;
uint32_t diff_len = 0; //length diff between original and new
uint32_t finalseq = 0; //to keep track of the new seq number
bool first = true;
bool end;
bool new = false;

struct rte_mempool *mod_packet_pool; //modified packet pool
struct rte_mbuf* ack_packet;
struct rte_mbuf* data_packet;
struct rte_mbuf* last_packet;
struct rte_mbuf *ackpacket;

/* To contain packet references of data packets*/
struct PacketArray {
    struct rte_mbuf **packets;
    size_t size;
    size_t capacity;
};
struct PacketArray packet_array;


struct flow_info {
	uint32_t	sig_id;				/* Signature id that was matched for this flow */
	uint64_t	scanned_bytes[2];		/* Scanned bytes for each direction */
	uint8_t		state;				/* State of the flow (SFT state) */
	struct doca_dpi_flow_ctx *dpi_flow_ctx;		/* DPI flow context */
	struct doca_telemetry_netflow_record record[2];	/* 1 - initiator, 0 - Receiver */
};

/* Per worker context */
struct worker_ctx {
	uint8_t		queue_id;				/* Queue id */
	uint16_t	ingress_port;				/* Current ingress port */
	uint64_t	dropped_packets;			/* Packets that failed to transmit */
	uint64_t	processed_packets;			/* Packets that were processed by this worker */
	struct		dpi_worker_attr attr;			/* DPI attributes */
	struct rte_eth_dev_tx_buffer *tx_buffer[SFT_PORTS_NUM];	/* Transmit buffers */
	struct doca_workq *workq;				/* The workq attached to dpi_ctx */
	struct rte_mempool *meta_mempool;			/* The pre-allocated job meta data buffer */
	struct rte_mempool *mod_packet_pool; //pool for modified buckets
	struct doca_buf_inventory *buf_inventory;		/* DOCA buf_inventory for dpdk_bridge */
	struct doca_dpdk_mempool *doca_dpdk_pool;		/* DOCA doca_buf pool for dpdk_bridge */
};



/* A helper struct for sending a job */
struct dpi_job_meta_data {
	struct flow_info *flow;		/* The flow context for a DPI job */
	struct rte_mbuf *mbuf;		/* The rte_mbuf intended to send by a DPI job */
	struct doca_dpi_result result;	/* The result for a DPI job */
};

/* DOCA core objects used by the samples
Should be renamed to compress core objects/ applications */
struct program_core_objects {
	struct doca_dev *dev;			/* doca device */
	//struct doca_mmap *src_mmap;		/* doca mmap for source buffer */
	struct doca_mmap *dst_mmap;		/* doca mmap for destination buffer */
	struct doca_buf_inventory *buf_inv;	/* doca buffer inventory */
	//struct doca_ctx *ctx;			/* doca context */
	struct doca_workq *workq;		/* doca work queue */
};

struct message {
    char data[10]; //arbitrary number
};

static inline void
add_ether_hdr(struct rte_mbuf *pkt_src, struct rte_mbuf *pkt_dst)
{
    struct rte_ether_hdr *eth_from;
    struct rte_ether_hdr *eth_to;
    eth_from = rte_pktmbuf_mtod(pkt_src, struct rte_ether_hdr *);
    eth_to = rte_pktmbuf_mtod(pkt_dst, struct rte_ether_hdr *);
    /* copy header */
    rte_memcpy(eth_to, eth_from, sizeof(struct rte_ether_hdr));
}

/*Check IP destination*/
bool get_src(struct rte_mbuf *pkt) {
    struct rte_ipv4_hdr *ipv4_hdr;
    ipv4_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));

    // Check if the packet is IPv4
    if (ipv4_hdr->version_ihl >> 4 == 4) {
        // IPv4 packet
        uint32_t src_ip = rte_be_to_cpu_32(ipv4_hdr->src_addr);
		uint32_t expected_ip = MAKE_IPV4_ADDRESS(10, 10, 10, 111);
       
		if (src_ip == expected_ip) {
		    //printf("Source IP address: %u.%u.%u.%u\n", 
           // (src_ip >> 24) & 0xFF, (src_ip >> 16) & 0xFF, (src_ip >> 8) & 0xFF, src_ip & 0xFF);
 
			return true;
	    }
	}

	return false;

}


static uint16_t
get_psd_sum(void *l3_hdr, uint16_t ethertype, uint64_t ol_flags)
{
        if (ethertype ==  RTE_ETHER_TYPE_IPV4)
                return rte_ipv4_phdr_cksum(l3_hdr, ol_flags);
        else /* assume ethertype == ETHER_TYPE_IPv6 */
                return rte_ipv6_phdr_cksum(l3_hdr, ol_flags);
}

static uint64_t
process_inner_cksums(struct rte_ether_hdr *eth_hdr, struct rte_net_hdr_lens *info)
{
        void *l3_hdr = NULL;
        uint8_t l4_proto;
        uint16_t ethertype;
        struct rte_ipv4_hdr *ipv4_hdr;
        struct rte_ipv6_hdr *ipv6_hdr;
        struct rte_udp_hdr *udp_hdr;
        struct rte_tcp_hdr *tcp_hdr;
        struct rte_sctp_hdr *sctp_hdr;
        uint64_t ol_flags = 0;
        info->l2_len = sizeof(struct rte_ether_hdr);
        ethertype = rte_be_to_cpu_16(eth_hdr->ether_type);
        /*if (ethertype == ETHER_TYPE_VLAN) {
                struct vlan_hdr *vlan_hdr = (struct vlan_hdr *)(eth_hdr + 1);
                info->l2_len  += sizeof(struct vlan_hdr);
                ethertype = rte_be_to_cpu_16(vlan_hdr->eth_proto);
        }*/
        l3_hdr = (char *)eth_hdr + info->l2_len;
        if (ethertype ==  RTE_ETHER_TYPE_IPV4) {
                ipv4_hdr = (struct ipv4_hdr *)l3_hdr;
                ipv4_hdr->hdr_checksum = 0;
                ol_flags |= PKT_TX_IPV4;
                ol_flags |= PKT_TX_IP_CKSUM;
                info->l3_len = sizeof(struct rte_ipv4_hdr);
                l4_proto = ipv4_hdr->next_proto_id;
        } else if (ethertype ==  RTE_ETHER_TYPE_IPV6) {
                ipv6_hdr = (struct ipv6_hdr *)l3_hdr;
                info->l3_len = sizeof(struct rte_ipv6_hdr);
                l4_proto = ipv6_hdr->proto;
                ol_flags |= PKT_TX_IPV6;
        } else
                return 0; /* packet type not supported, nothing to do */
        if (l4_proto == IPPROTO_UDP) {
                udp_hdr = (struct udp_hdr *)((char *)l3_hdr + info->l3_len);
                ol_flags |= PKT_TX_UDP_CKSUM;
				udp_hdr->dgram_cksum = 0;
                udp_hdr->dgram_cksum = get_psd_sum(l3_hdr,
                                ethertype, ol_flags);
        } else if (l4_proto == IPPROTO_TCP) {
			    
                tcp_hdr = (struct tcp_hdr *)((char *)l3_hdr + info->l3_len);
                /* Put PKT_TX_TCP_SEG bit setting before get_psd_sum(), because
                 * it depends on PKT_TX_TCP_SEG to calculate pseudo-header
                 * checksum.
                 */
                /*if (tso_segsz != 0) {
                        ol_flags |= PKT_TX_TCP_SEG;
                        info->tso_segsz = tso_segsz;
                        info->l4_len = (tcp_hdr->data_off & 0xf0) >> 2;
                }*/
				tcp_hdr->cksum = 0;
				ol_flags |= RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM;
                tcp_hdr->cksum = get_psd_sum(l3_hdr, ethertype, ol_flags);
        } 
        return ol_flags;
}

static void checksum_calculate(struct rte_mbuf *new_packet)
{

	struct rte_ipv4_hdr *ipv4_hdr;
	struct rte_ether_hdr *eth_hdr;
	struct rte_udp_hdr *udp;
    struct rte_net_hdr_lens tx_offload;
    uint32_t ptype;
	uint64_t ol_flags = 0;
	uint32_t old_len = new_packet->pkt_len, hash;

	eth_hdr = rte_pktmbuf_mtod(new_packet, struct rte_ether_hdr *);
	ipv4_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
	//tcp_hdr = rte_pktmbuf_mtod_offset(new_packet, struct rte_tcp_hdr *, new_packet->l2_len + new_packet->l3_len);
    udp = (struct rte_udp_hdr *)(ipv4_hdr + 1);

    
    ptype = rte_net_get_ptype(new_packet, &tx_offload, RTE_PTYPE_ALL_MASK);
	ipv4_hdr->total_length = rte_cpu_to_be_16(new_packet->pkt_len - sizeof(*eth_hdr));

    void *l3_hdr = NULL;
    l3_hdr = (char *)eth_hdr + tx_offload.l2_len;

/* outer IP checksum */

	ol_flags |= RTE_MBUF_F_TX_OUTER_IP_CKSUM;
	ipv4_hdr->hdr_checksum = 0;


/* inner checksum */
    //new_packet->ol_flags |= RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM;
	ol_flags |= process_inner_cksums(eth_hdr, &tx_offload);

	new_packet->l2_len = tx_offload.l2_len;
	new_packet->l3_len = tx_offload.l3_len;
	new_packet->l4_len = tx_offload.l4_len;

	new_packet->outer_l2_len = sizeof(struct rte_ether_hdr);
	new_packet->outer_l3_len = sizeof(struct rte_ipv4_hdr);



/* outer IP checksum */
	ipv4_hdr->hdr_checksum = rte_ipv4_cksum(ipv4_hdr);  
	
}

static inline void
update_pkt_header(struct rte_mbuf *pkt, uint32_t total_pkt_len)
{
	struct rte_ipv4_hdr *ip_hdr;
	struct rte_tcp_hdr *tcp_hdr;
	uint16_t pkt_data_len;
	uint16_t pkt_len;

	pkt_data_len = (uint16_t) (total_pkt_len - (
					sizeof(struct rte_ether_hdr) +
					sizeof(struct rte_ipv4_hdr) +
					sizeof(struct rte_tcp_hdr)));

	//printf("Pkt data len%d \n", pkt_data_len);
	/* update UDP packet length */
	tcp_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_tcp_hdr *,
				sizeof(struct rte_ether_hdr) +
				sizeof(struct rte_ipv4_hdr));
	pkt_len = (uint16_t) (pkt_data_len + sizeof(struct rte_tcp_hdr));
	//udp_hdr->dgram_len = rte_cpu_to_be_16(pkt_len);

	//udp_hdr->dgram_cksum = 0;
	//tcp_hdr->cksum = 0;

	/* update IP packet length and checksum */
	ip_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_ipv4_hdr *,
				sizeof(struct rte_ether_hdr));
	ip_hdr->hdr_checksum = 0;
	pkt_len = (uint16_t) (pkt_len + sizeof(struct rte_ipv4_hdr));
	ip_hdr->total_length = rte_cpu_to_be_16(pkt_len);
	//ip_hdr->hdr_checksum = rte_ipv4_cksum(ip_hdr);
}


void print_tcp_header_info(struct rte_mbuf *pkt) {
    struct rte_ipv4_hdr *ipv4_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));
    struct tcphdr *tcp_hdr = (struct tcphdr *)((char *)ipv4_hdr + sizeof(struct rte_ipv4_hdr));

    printf("Source Port: %u\n", rte_be_to_cpu_16(tcp_hdr->th_sport));
    printf("Destination Port: %u\n", rte_be_to_cpu_16(tcp_hdr->th_dport));
    /*printf("Sequence Number: %u\n", rte_be_to_cpu_32(tcp_hdr->th_seq));
    printf("Acknowledgment Number: %u\n", rte_be_to_cpu_32(tcp_hdr->th_ack));
    printf("Data Offset: %u bytes\n", tcp_hdr->th_off * 4); // th_off is the number of 32-bit words, so we multiply by 4 to get bytes
    printf("Flags:\n");
    if (tcp_hdr->th_flags & TH_FIN)
        printf("FIN\n");
    if (tcp_hdr->th_flags & TH_SYN)
        printf("SYN\n");
    if (tcp_hdr->th_flags & TH_RST)
        printf("RST\n");
    if (tcp_hdr->th_flags & TH_PUSH)
        printf("PSH\n");
    if (tcp_hdr->th_flags & TH_ACK)
        printf("ACK\n");
    if (tcp_hdr->th_flags & TH_URG)
        printf("URG\n");
    printf("Window Size: %u\n", rte_be_to_cpu_16(tcp_hdr->th_win));
    printf("Checksum: %u\n", rte_be_to_cpu_16(tcp_hdr->th_sum));
    printf("Urgent Pointer: %u\n", rte_be_to_cpu_16(tcp_hdr->th_urp));*/
}


/**
 * Check if given device is capable of executing a DOCA_COMPRESS_DEFLATE_JOB.
 *
 * @devinfo [in]: The DOCA device information
 * @return: DOCA_SUCCESS if the device supports DOCA_COMPRESS_DEFLATE_JOB and DOCA_ERROR otherwise.
 */
static doca_error_t
compress_jobs_compress_is_supported(struct doca_devinfo *devinfo)
{
	return doca_compress_job_get_supported(devinfo, DOCA_COMPRESS_DEFLATE_JOB);
}

doca_error_t
dpi_job_is_supported(struct doca_devinfo *devinfo)
{
	return doca_dpi_job_get_supported(devinfo, DOCA_DPI_JOB);
}

doca_error_t
doca_dpi_worker_ctx_create(struct doca_dpi_worker_ctx **dpi_ctx,
			   uint16_t max_sig_match_len,
			   uint32_t per_workq_packet_pool_size,
			   struct doca_dev *dev,
			   const char *sig_file_path)
{
	doca_error_t result;
	struct doca_dpi_worker_ctx *ctx = NULL;

	if (dpi_ctx == NULL || dev == NULL)
		return DOCA_ERROR_INVALID_VALUE;

	ctx = (struct doca_dpi_worker_ctx *)calloc(1, sizeof(struct doca_dpi_worker_ctx));
	if (ctx == NULL) {
		DOCA_LOG_ERR("Failure to allocate doca_dpi_worker!");
		*dpi_ctx = NULL;
		return DOCA_ERROR_NO_MEMORY;
	}

	/* Create doca_dpi instance */
	result = doca_dpi_create(&ctx->dpi);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create doca_dpi instance. err = [%s]",
				doca_get_error_string(result));
		goto dpi_create_failure;
	}

	/* Add doca_dev into doca_dpi instance */
	ctx->dev = dev;
	result = doca_ctx_dev_add(doca_dpi_as_ctx(ctx->dpi), ctx->dev);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to register device with dpi. err = [%s]",
				doca_get_error_string(result));
		goto dpi_destroy;
	}

	/* Load signatures into doca_dpi's backend device */
	if (sig_file_path != NULL) {
		result = doca_dpi_set_signatures(ctx->dpi, sig_file_path);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Loading DPI signatures failed, err = [%s]",
					doca_get_error_string(result));
			goto dpi_destroy;
		}
	}

	/* Set workq packet pool size of each workq */
	result = doca_dpi_set_per_workq_packet_pool_size(ctx->dpi, per_workq_packet_pool_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Set per_workq_packet_pool_size failed, err = [%s]",
				doca_get_error_string(result));
		goto dpi_destroy;
	}

	/* Set workq max num of flows for each workq
	 * this num is set according to max num of packets so each flow can have enough packets
	 * for sig identification
	 */
	result = doca_dpi_set_per_workq_max_flows(ctx->dpi, per_workq_packet_pool_size/MAX_MBUFS_FOR_FLOW);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Set per_workq_packet_pool_size failed, err = [%s]",
				doca_get_error_string(result));
		goto dpi_destroy;
	}

	/* Set max_sig_match_len */
	result = doca_dpi_set_max_sig_match_len(ctx->dpi, max_sig_match_len);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Set max_sig_match_len failed, err = [%s]",
				doca_get_error_string(result));
		goto dpi_destroy;
	}

	/* Start doca_dpi */
	result = doca_ctx_start(doca_dpi_as_ctx(ctx->dpi));
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to start doca_dpi, err = [%s]",
				doca_get_error_string(result));
		goto dpi_destroy;
	}

	*dpi_ctx = ctx;
	return result;

dpi_destroy:
	doca_ctx_stop(doca_dpi_as_ctx(ctx->dpi));
	doca_dpi_destroy(ctx->dpi);
	ctx->dpi = NULL;
	doca_dev_close(ctx->dev);
	ctx->dev = NULL;
dpi_create_failure:
	free(ctx);
	*dpi_ctx = NULL;
	return result;
}

void
doca_dpi_worker_ctx_destroy(struct doca_dpi_worker_ctx *ctx)
{
	if (ctx == NULL)
		return;

	if (ctx->dpi != NULL) {
		doca_ctx_stop(doca_dpi_as_ctx(ctx->dpi));
		doca_dpi_destroy(ctx->dpi);
		ctx->dpi = NULL;
	}
	if (ctx->dev != NULL) {
		doca_dev_close(ctx->dev);
		ctx->dev = NULL;
	}
	free(ctx);
}

/*
 * Calculates the L7 offset for the given packet.
 *
 * @packet [in]: Packet to calculate the L7 offset for.
 * @return: L7 offset.
 */
static DPI_WORKER_ALWAYS_INLINE uint32_t
get_payload_offset(const struct rte_mbuf *packet)
{
	struct rte_net_hdr_lens headers = {0};

	rte_net_get_ptype(packet, &headers, RTE_PTYPE_ALL_MASK);
	return headers.l2_len + headers.l3_len + headers.l4_len;
}

/*
 * Try to send netflow record to the netflow buffer
 *
 * @flow [in]: Flow to send
 * @ctx [in]: Worker context
 * @initiator [in]: 0x1 if the flow is initiator, 0x0 otherwise
 */
static DPI_WORKER_ALWAYS_INLINE void
set_netflow_record(struct flow_info *flow, const struct worker_ctx *ctx, const uint8_t initiator)
{
	struct doca_telemetry_netflow_record *record_to_send;

	if (ctx->attr.send_netflow_record == NULL)
		return;

	record_to_send = &flow->record[!!initiator];
	record_to_send->last = time(0);
	ctx->attr.send_netflow_record(record_to_send);
	/* Only the difference is relevant between Netflow interactions */
	record_to_send->d_pkts = 0;
	record_to_send->d_octets = 0;
}

/*
 * The reverse_stpl takes a 7 tuple as an input and reverses it.
 * 5-tuple reversal is ordinary while the zone stays the same for both
 * directions. The last piece of the 7-tuple is the port which is also reversed.
 *
 * @stpl [in]: 7-tuple to reverse
 * @rstpl [out]: Reversed 7-tuple
 */
static DPI_WORKER_ALWAYS_INLINE void
reverse_stpl(const struct rte_sft_7tuple *stpl, struct rte_sft_7tuple *rstpl)
{
	memset(rstpl, 0, sizeof(*rstpl));
	rstpl->flow_5tuple.is_ipv6 = stpl->flow_5tuple.is_ipv6;
	rstpl->flow_5tuple.proto = stpl->flow_5tuple.proto;
	if (rstpl->flow_5tuple.is_ipv6) {
		memcpy(&rstpl->flow_5tuple.ipv6.src_addr[0], &stpl->flow_5tuple.ipv6.dst_addr[0], 16);
		memcpy(&rstpl->flow_5tuple.ipv6.dst_addr[0], &stpl->flow_5tuple.ipv6.src_addr[0], 16);
	} else {
		rstpl->flow_5tuple.ipv4.src_addr = stpl->flow_5tuple.ipv4.dst_addr;
		rstpl->flow_5tuple.ipv4.dst_addr = stpl->flow_5tuple.ipv4.src_addr;
	}
	rstpl->flow_5tuple.src_port = stpl->flow_5tuple.dst_port;
	rstpl->flow_5tuple.dst_port = stpl->flow_5tuple.src_port;
	rstpl->zone = stpl->zone;
	rstpl->port_id = stpl->port_id ^ 1;
}

/*
 * Set L4 fields needed by DPI
 *
 * @mbuf_info [in]: mbuf info to parse from
 * @parsing_info [in]: Parsing info to set L4 fields for
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static DPI_WORKER_ALWAYS_INLINE doca_error_t
set_l4_parsing_info(const struct rte_sft_mbuf_info *mbuf_info, struct doca_dpi_parsing_info *parsing_info)
{
	parsing_info->ethertype = rte_cpu_to_be_16(mbuf_info->eth_type);
	parsing_info->l4_protocol = mbuf_info->l4_protocol;

	if (!mbuf_info->is_ipv6)
		parsing_info->dst_ip.ipv4.s_addr = mbuf_info->ip4->dst_addr;
	else
		memcpy(&parsing_info->dst_ip.ipv6, &mbuf_info->ip6->dst_addr[0], 16);
	if (parsing_info->l4_protocol == IPPROTO_UDP) {
		parsing_info->l4_sport = mbuf_info->udp->src_port;
		parsing_info->l4_dport = mbuf_info->udp->dst_port;
	} else if (parsing_info->l4_protocol == IPPROTO_TCP) {
		parsing_info->l4_sport = mbuf_info->tcp->src_port;
		parsing_info->l4_dport = mbuf_info->tcp->dst_port;
	} else {
		DOCA_DLOG_DBG("Unsupported L4 protocol!");
		return DOCA_ERROR_NOT_SUPPORTED;
	}
	return DOCA_SUCCESS;
}

/*
 * Initialize the flow_info structure to be associated with the flow
 *
 * @fid [in]: Flow ID to initialize
 * @dpi_flow_ctx [in]: DPI flow context to initialize the flow with
 * @ctx [in]: Worker context
 * @five_tuple [in]: 5-tuple of the flow
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
client_obj_flow_info_create(uint32_t fid,
			struct doca_dpi_flow_ctx *dpi_flow_ctx,
			const struct worker_ctx *ctx,
			const struct rte_sft_5tuple *five_tuple)
{
	struct flow_info *flow = (struct flow_info *)rte_zmalloc(NULL, sizeof(*flow), 0);
	struct rte_sft_error error;

	if (flow == NULL)
		return DOCA_ERROR_NO_MEMORY;

	strcpy(flow->record[INITIATOR].application_name, "NO_MATCH");
	flow->record[INITIATOR].flow_id = fid;
	if (!five_tuple->is_ipv6) {
		flow->record[INITIATOR].src_addr_v4 = five_tuple->ipv4.src_addr;
		flow->record[INITIATOR].dst_addr_v4 = five_tuple->ipv4.dst_addr;
	} else {
		memcpy(&flow->record[INITIATOR].src_addr_v6, five_tuple->ipv6.src_addr, 16);
		memcpy(&flow->record[INITIATOR].dst_addr_v6, five_tuple->ipv6.dst_addr, 16);
	}
	flow->record[INITIATOR].src_port = five_tuple->src_port;
	flow->record[INITIATOR].dst_port = five_tuple->dst_port;
	flow->record[INITIATOR].protocol = five_tuple->proto;
	flow->record[INITIATOR].input = ctx->ingress_port;
	flow->record[INITIATOR].output = ctx->ingress_port ^ 1;
	flow->record[INITIATOR].first = time(0);
	flow->record[INITIATOR].last = time(0);

	strcpy(flow->record[RECEIVER].application_name, "NO_MATCH");
	flow->record[RECEIVER].flow_id = fid;
	if (!five_tuple->is_ipv6) {
		flow->record[RECEIVER].src_addr_v4 = five_tuple->ipv4.dst_addr;
		flow->record[RECEIVER].dst_addr_v4 = five_tuple->ipv4.src_addr;
	} else {
		memcpy(&flow->record[RECEIVER].src_addr_v6, five_tuple->ipv6.dst_addr, 16);
		memcpy(&flow->record[RECEIVER].dst_addr_v6, five_tuple->ipv6.src_addr, 16);
	}
	flow->record[RECEIVER].src_port = five_tuple->dst_port;
	flow->record[RECEIVER].dst_port = five_tuple->src_port;
	flow->record[RECEIVER].protocol = five_tuple->proto;
	flow->record[RECEIVER].input = ctx->ingress_port ^ 1;
	flow->record[RECEIVER].output = ctx->ingress_port;
	flow->record[RECEIVER].first = time(0);
	flow->record[RECEIVER].last = time(0);

	if (rte_sft_flow_set_client_obj(ctx->queue_id, fid, SFT_FLOW_INFO_CLIENT_ID, flow, &error) != 0) {
		rte_free(flow);
		return DOCA_ERROR_DRIVER;
	}
	flow->dpi_flow_ctx = dpi_flow_ctx;
	set_netflow_record(flow, ctx, INITIATOR); /* First packet is initiator */
	return DOCA_SUCCESS;
}

/*
 * Update the flow_info structure with signature name that was matched
 *
 * @app_name [in]: Signature name that was matched
 * @sig_id [in]: Signature ID
 * @data [in]: Flow info structure to update
 */
static void
client_obj_flow_info_set(const char *app_name, uint32_t sig_id, struct flow_info *data)
{
	assert(data != NULL);
	data->sig_id = sig_id;
	memcpy(data->record[0].application_name, app_name, 64);
	memcpy(data->record[1].application_name, app_name, 64);
}

/*
 * Get SFT state according to the user-defined function
 *
 * @ctx [in]: Worker context
 * @dpi_result [in]: DPI result containing match information
 * @fid [in]: Flow ID of the matched flow
 * @return: SFT state
 */
static enum SFT_USER_STATE
get_sft_state_from_match(const struct worker_ctx *ctx, const struct doca_dpi_result *dpi_result, uint32_t fid)
{
	enum dpi_worker_action dpi_action;

	if (ctx->attr.dpi_on_match == NULL)
		return RSS_FLOW;

	if (ctx->attr.dpi_on_match(dpi_result, fid, ctx->attr.user_data, &dpi_action) != 0)
		return RSS_FLOW;
	switch (dpi_action) {
	case DPI_WORKER_ALLOW:
		return HAIRPIN_MATCHED_FLOW;
	case DPI_WORKER_DROP:
		return DROP_FLOW;
	default:
		return RSS_FLOW;
	}
}

/*
 * Update Netflow record counters
 *
 * @ctx [in]: Worker context
 * @flow [in]: Flow info structure to update
 * @packet [in]: Packet that processed by the worker
 * @initiator [in]: 0x1 for initiator, 0x0 for RECEIVER
 */
static void
update_record_counters(const struct worker_ctx *ctx, struct flow_info *flow,
			const struct rte_mbuf *packet, const uint8_t initiator)
{
	flow->record[!!initiator].d_pkts++;
	flow->record[!!initiator].d_octets += rte_pktmbuf_pkt_len(packet);

	/* Every predefined number of packets, we send a Netflow record */
	if (flow->record[initiator].d_pkts % NETFLOW_UPDATE_RATE == 0)
		set_netflow_record(flow, ctx, initiator);
}

/*
 * Called on DPI match, relevant flow's state is updated and Netflow records are sent
 *
 * @flow [in]: Flow info structure associated with the matched flow
 * @result [in]: DPI result containing match information
 * @ctx [in]: Worker context
 */
static void
resolve_dpi_match(struct flow_info *flow, const struct doca_dpi_result *result, const struct worker_ctx *ctx)
{
	uint32_t fid = flow->record[RECEIVER].flow_id;
	struct rte_sft_error error;
	struct doca_dpi_sig_data sig_data = {0};

	DOCA_DLOG_DBG("FID %u matches sig_id %d", fid, result->info.sig_id);

	flow->state =  get_sft_state_from_match(ctx, result, fid);

	if (rte_sft_flow_set_state(ctx->queue_id, fid, flow->state, &error) != 0)
		return;

	if (doca_dpi_get_signature(ctx->attr.dpi_ctx->dpi, result->info.sig_id, &sig_data) != DOCA_SUCCESS)
		return;

	client_obj_flow_info_set(sig_data.name, result->info.sig_id, flow);
	/* Update match for both Netflow directions */
	set_netflow_record(flow, ctx, INITIATOR);
	set_netflow_record(flow, ctx, RECEIVER);
}

/*
 * Destroys DPI flow context
 *
 * @flow [in]: Flow info structure to destroy
 * @ctx [in]: Worker context
 */
static void
resolve_dpi_destroy(struct flow_info *flow, const struct worker_ctx *ctx)
{
	set_netflow_record(flow, ctx, INITIATOR);
	set_netflow_record(flow, ctx, RECEIVER);
	/* In some scenarios it is possible to have a SFT flow without DPI flow */
	if (flow->dpi_flow_ctx == NULL)
		return;
	doca_dpi_flow_destroy(flow->dpi_flow_ctx);
	DOCA_DLOG_DBG("DPI FID %llu was destroyed", flow->record[0].flow_id);
	flow->dpi_flow_ctx = NULL;
	rte_free(flow);
}

/*
 * Retrieve and destroy aged flows
 *
 * @ctx [in]: Worker context
 */
static void
clear_aged_flows(const struct worker_ctx *ctx)
{
	int aged_flows, fid_index;
	uint32_t fid;
	uint32_t *fid_list = NULL;
	struct flow_info *flow = NULL;
	struct rte_sft_error error;
	/* if nb_fids is 0, return the number of all aged out SFT flows */
	aged_flows = rte_sft_flow_get_aged_flows(ctx->queue_id, fid_list, /* nb_fids */ 0, &error);
	if (aged_flows <= 0)
		return;
	fid_list = (uint32_t *)rte_zmalloc(NULL, sizeof(uint32_t) * aged_flows, 0);
	if (unlikely(fid_list == NULL))
		return;
	/* if nb_fids is not 0 , return the number of aged out flows - IT HAS TO BE EQUAL */
	if (rte_sft_flow_get_aged_flows(ctx->queue_id, fid_list, aged_flows, &error) < 0)
		return;
	for (fid_index = 0; fid_index < aged_flows; fid_index++) {
		fid = fid_list[fid_index];
		DOCA_DLOG_DBG("FID %u will be removed due to aging", fid);
		flow = (struct flow_info *)rte_sft_flow_get_client_obj(ctx->queue_id, fid, SFT_FLOW_INFO_CLIENT_ID, &error);
		assert(flow != NULL);
		resolve_dpi_destroy(flow, ctx);
		if (rte_sft_flow_destroy(ctx->queue_id, fid, &error) != 0)
			DOCA_LOG_ERR("FID %u destroy failed", fid);
	}
	rte_free(fid_list);
}

/*
 * Phase 2 of flow creation, this function should be called only after rte_sft_process_mbuf()
 *
 * @packet [in]: Packet to process
 * @ctx [in]: Worker context
 * @sft_status [in/out]: SFT state of the flow
 * @sft_packet [out]: Packet received by the SFT
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t
activate_new_connection(struct rte_mbuf *packet,
			const struct worker_ctx *ctx,
			struct rte_sft_flow_status *sft_status,
			struct rte_mbuf **sft_packet)
{
	int ret;
	struct doca_dpi_flow_ctx *dpi_flow_ctx;
	struct rte_sft_7tuple stpl, rstpl;
	struct rte_sft_error error;
	struct rte_sft_mbuf_info mbuf_info = {0};
	struct doca_dpi_parsing_info parsing_info = {0};
	uint32_t data = 0xdeadbeef;
	const uint8_t queue_id = ctx->queue_id;
	const uint16_t port_id = ctx->ingress_port;
	const uint32_t default_state = 0;
	const uint8_t device_id = 0;
	const uint8_t enable_proto_state = 1;
	const struct rte_sft_actions_specs sft_action = {
						.actions = RTE_SFT_ACTION_AGE | RTE_SFT_ACTION_COUNT,
						.initiator_nat = NULL,
						.reverse_nat = NULL,
						.aging = 0};
	doca_error_t result;
	struct doca_workq *workq = ctx->workq;

	if (unlikely(!sft_status->zone_valid)) {
		ret = rte_sft_process_mbuf_with_zone(queue_id, packet, SFT_ZONE, sft_packet, sft_status, &error);
		if (unlikely(ret < 0))
			return DOCA_ERROR_DRIVER;
	}

	if (!sft_status->activated) {
		ret = rte_sft_parse_mbuf(packet, &mbuf_info, NULL, &error);
		if (unlikely(ret < 0))
			return DOCA_ERROR_DRIVER;
		rte_sft_mbuf_stpl(packet, &mbuf_info, sft_status->zone, &stpl, &error);
		reverse_stpl(&stpl, &rstpl);
		ret = rte_sft_flow_activate(queue_id,
					    SFT_ZONE,		/* Fixed zone */
					    packet,
					    &rstpl,
					    default_state,	/* Default state = 0 */
					    &data,
					    enable_proto_state, /* always maintain protocol state */
					    &sft_action,
					    device_id,
					    port_id,
					    sft_packet,
					    sft_status,
					    &error);
		if (unlikely(ret < 0)) {
			rte_pktmbuf_free(packet);
			return DOCA_ERROR_DRIVER;
		}

		/* Flow activated at this point */
		assert(sft_status->activated);
		if (unlikely(*sft_packet == NULL)) {
			result = DOCA_ERROR_DRIVER;
			goto flow_destroy;
		}
		DOCA_DLOG_DBG("Flow activated");

		if (unlikely(sft_status->proto_state == SFT_CT_STATE_ERROR)) {
			result = DOCA_ERROR_DRIVER;
			goto flow_destroy;
		}
		if (rte_sft_flow_get_client_obj(queue_id, sft_status->fid, SFT_FLOW_INFO_CLIENT_ID, &error) == NULL) {
			result = set_l4_parsing_info(&mbuf_info, &parsing_info);
			if (unlikely(result != DOCA_SUCCESS))
				goto flow_destroy;

			result = doca_dpi_flow_create(ctx->attr.dpi_ctx->dpi, workq,
						&parsing_info, &dpi_flow_ctx);
			if (unlikely(result != DOCA_SUCCESS))
				goto flow_destroy;

			result = client_obj_flow_info_create(sft_status->fid, dpi_flow_ctx, ctx, &stpl.flow_5tuple);

			if (unlikely(result != DOCA_SUCCESS)) {
				doca_dpi_flow_destroy(dpi_flow_ctx);
				goto flow_destroy;
			}
		}
		DOCA_DLOG_DBG("New flow activated (fid=%u)", sft_status->fid);
	}
	return DOCA_SUCCESS;

flow_destroy:
	if (rte_sft_flow_destroy(queue_id, sft_status->fid, &error) != 0)
		DOCA_LOG_ERR("FID %u destroy failed", sft_status->fid);
	return result;
}

/*
 * Dequeue packets from the DPI engine till the queue is empty
 * Each packet dequeued is checked for signature matches, if any, the flow's state is updated
 * Packets which match 'DROP' signature are dropped
 * All other packets are buffered for later transmission
 *
 * @ctx [in]: Worker context
 */
static void
dequeue_from_dpi(const struct worker_ctx *ctx)
{
	doca_error_t result;
	struct doca_dpi_result *res = NULL;
	struct doca_event event = {0};
	struct doca_workq *workq = ctx->workq;
	struct rte_mempool *mempool = ctx->meta_mempool;
	struct rte_mbuf *pkt;
	struct dpi_job_meta_data *job_meta;

	result = doca_workq_progress_retrieve(workq, &event, DOCA_WORKQ_RETRIEVE_FLAGS_NONE);

	while (result == DOCA_SUCCESS) {
		res = (struct doca_dpi_result *)(event.result.ptr);
		job_meta = (struct dpi_job_meta_data *)(event.user_data.ptr);
		pkt = job_meta->mbuf;
		if (unlikely(res->status_flags & DOCA_DPI_STATUS_DESTROYED || event.user_data.ptr == NULL))
			goto skip_packet;
		if (likely(!res->matched))
			TX_BUFFER_PKT(pkt, ctx);
		else {
			if (res->info.action != DOCA_DPI_SIG_ACTION_DROP)
				TX_BUFFER_PKT(pkt, ctx);
			if (res->status_flags & DOCA_DPI_STATUS_NEW_MATCH)
				resolve_dpi_match(job_meta->flow, res, ctx);
		}
skip_packet:
		doca_buf_refcount_rm((struct doca_buf *)(res->pkt), NULL);
		rte_mempool_put(mempool, (void *)job_meta);
		result = doca_workq_progress_retrieve(workq, &event, DOCA_WORKQ_RETRIEVE_FLAGS_NONE);
	}
}




/*
 * Enqueue a packet to the DPI engine
 * Each enqueued packet is checked for signature matches and results are retrieved in dequeue_from_dpi()
 * Empty packets are not enqueued and buffered for later transmission
 * If the DPI signatures are not loaded to DPI engine, no packets are enqueued and buffered for later transmission
 *
 * @sft_packet [in]: Packet to be enqueued
 * @flow [in]: Flow info associated with the packet's flow
 * @sft_status [in]: SFT status associated with the flow
 * @ctx [in]: Worker context
 */
static void
enqueue_packet_to_dpi(struct rte_mbuf *sft_packet,
		      struct flow_info *flow,
		      const struct rte_sft_flow_status *sft_status,
		      const struct worker_ctx *ctx)
{
	doca_error_t result;
	uint32_t payload_offset;
	const uint64_t max_dpi_depth = ctx->attr.max_dpi_depth;
	struct rte_sft_error sft_error;
	struct rte_mempool *mempool = ctx->meta_mempool;
	struct doca_workq *workq = ctx->workq;
	struct dpi_job_meta_data *job_meta = NULL;
	struct doca_buf *pkt_doca_buf = NULL;
	/*mod*/
	struct doca_buf *dst_doca_buf;
	 struct rte_net_hdr_lens tx_offload;


    unsigned char* new_payload = "5";

	struct rte_sft_error error;
	struct rte_sft_mbuf_info mbuf_info;

	char *data;
	uint16_t payload_length;
	uint32_t data_offset = 0;
    int len;
	uint16_t pkt_data_len;
	uint16_t pkt_len;
	uint16_t header_len;
	struct rte_mbuf* new_packet = NULL;


	if (unlikely(flow->dpi_flow_ctx == NULL)) {
		rte_pktmbuf_free(sft_packet);
		return;
	}

	payload_offset = get_payload_offset(sft_packet);

	bool src_flag = get_src(sft_packet);
	uint16_t p_length = sft_packet->pkt_len - payload_offset;

 	
    /*printf("Data: ");
    for (int i = 0; i < p_length; i++) {
        printf("%c", pdata[i]);
    }
    printf("\n");*/



    //struct rte_mbuf* new_packet = rte_pktmbuf_clone(sft_packet, mod_packet_pool); //rte_pktmbuf_alloc(mod_packet_pool);//(attr.dpdk_config->mbuf_pool);
    /*struct rte_mbuf* new_packet = rte_pktmbuf_copy(sft_packet, mod_packet_pool, 0, sft_packet->pkt_len);
	if (new_packet == NULL) {
		printf("Alloc failed");
        return;
    }*/
	


header_len = sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_tcp_hdr);

header_len = payload_offset;





if (p_length > 0 && src_flag ){

	new_packet = rte_pktmbuf_copy(sft_packet, mod_packet_pool, 0, header_len);
	if (new_packet == NULL) {
		printf("Alloc failed");
        return;
    }else{
		sft_packet = new_packet;
	}

   
	// Copy the headers from the original packet to the new packet
    //rte_memcpy(rte_pktmbuf_mtod(new_packet, void*), rte_pktmbuf_mtod(sft_packet, void*), header_len);
    

	new_packet->pkt_len = strlen(new_payload) + header_len;
	new_packet->data_len = strlen(new_payload) + header_len;

    unsigned char* src = rte_pktmbuf_mtod_offset(new_packet, unsigned char*, payload_offset);  // Cast sft_packet to unsigned char* source pointer
    rte_memcpy(src, new_payload, strlen(new_payload));
	
	
	unsigned char* pdata = rte_pktmbuf_mtod_offset(new_packet, unsigned char*, payload_offset);

	uint16_t p_length = new_packet->pkt_len - payload_offset;

	//printf("Data:%d", payload_offset);
	//printf("Data:%d", sft_packet->pkt_len);
	//printf("Data:%d", new_packet->pkt_len);

	printf("Data:%d", p_length);
	printf("Data:%d\n", new_packet->pkt_len - payload_offset);
    for (int i = 0; i < p_length; i++) {
       printf("%c", pdata[i]);
    }
    printf("\n");

	printf("data len%d \n", new_packet->data_len);
	printf("Pkt data len%d \n", new_packet->pkt_len);

	
    struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(new_packet, struct rte_ether_hdr *);
    struct rte_ipv4_hdr *ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
    struct rte_tcp_hdr *tcp_hdr = (struct rte_tcp_hdr *)(ip_hdr + 1);



	uint32_t ptype;

    ptype = rte_net_get_ptype(new_packet, &tx_offload, RTE_PTYPE_ALL_MASK);
    
	/*This is not needed for non modified packet size*/
	update_pkt_header(new_packet, new_packet->pkt_len);

	new_packet->l2_len = sizeof(*eth_hdr);
    new_packet->l3_len = sizeof(*ip_hdr);
    ip_hdr->hdr_checksum = 0;
	tcp_hdr->cksum = 0;
    new_packet->ol_flags |= RTE_MBUF_F_TX_IPV4 |  RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM;

	printf("data len%d \n", new_packet->data_len);
	printf("Pkt data len%d \n", new_packet->pkt_len);
}


    struct rte_ether_hdr *eth_hdr1 = rte_pktmbuf_mtod(sft_packet, struct rte_ether_hdr *);
    struct rte_ipv4_hdr *ip_hdr1 = (struct rte_ipv4_hdr *)(eth_hdr1 + 1);
    struct rte_udp_hdr *tcp_hdr1 = (struct rte_tcp_hdr *)(ip_hdr1 + 1);

    print_tcp_header_info(sft_packet);
	if (new_packet!=NULL){
		printf("Related to new \n");
	}
	printf("\n"); 




/*End*/


	/*
	 * Try to get a meta buffer.
	 * If there is no meta buffer, call dequeue_from_dpi(), until we can get one.
	 */
	while (rte_mempool_get(mempool, (void **)&job_meta) != 0)
		dequeue_from_dpi(ctx);


	

	/* Convert rte_mbuf to doca_buf */
	result = doca_dpdk_mempool_mbuf_to_buf(ctx->doca_dpdk_pool, ctx->buf_inventory, sft_packet, &pkt_doca_buf);
	if (result != DOCA_SUCCESS) {
		rte_mempool_put(mempool, job_meta);
		return;
	}

	/* Record meta data */
	job_meta->flow = flow;
	job_meta->mbuf = sft_packet;

	/* Create a DPI job */
	struct doca_dpi_job job = (struct doca_dpi_job) {
		.base.type = DOCA_DPI_JOB,
		.base.flags = DOCA_JOB_FLAGS_NONE,
		.base.ctx = doca_dpi_as_ctx(ctx->attr.dpi_ctx->dpi),
		.base.user_data.ptr = (void *)job_meta,
		.pkt = pkt_doca_buf,
		.initiator = sft_status->initiator,
		.payload_offset = payload_offset,
		.flow_ctx = flow->dpi_flow_ctx,
		.result = &job_meta->result
	};

	result = doca_workq_submit(workq, &(job.base));
	if (result == DOCA_ERROR_INVALID_VALUE) {
		/* No signatures loaded */
		doca_buf_refcount_rm(pkt_doca_buf, NULL);
		rte_mempool_put(mempool, job_meta);
		goto forward_packet;
	}

	while (result == DOCA_ERROR_NO_MEMORY) { /* If the DPI is busy, dequeue until we successfully enqueue the packet */
		dequeue_from_dpi(ctx);
		result = doca_workq_submit(workq, &(job.base));
	}

	/* Netflow statistics */
	update_record_counters(ctx, flow, sft_packet, sft_status->initiator);
	if (result == DOCA_SUCCESS) {
		/* Update bytes counters */
		flow->scanned_bytes[sft_status->initiator] += (rte_pktmbuf_pkt_len(sft_packet) - payload_offset);
		/* When reaching max_dpi_depth, offload flow to HW */
		if (max_dpi_depth > 0 && flow->scanned_bytes[sft_status->initiator] > max_dpi_depth)
			if (rte_sft_flow_set_state(ctx->queue_id, sft_status->fid, HAIRPIN_SKIPPED_FLOW, &sft_error))
				DOCA_DLOG_DBG("Failed to set flow state (fid=%u)", sft_status->fid);

		return;
	}

	/* Empty packets will be forwarded */
forward_packet:
	TX_BUFFER_PKT(sft_packet, ctx);
}

/*
 * Drains fragmented packets from SFT according to the first packet in the fragmented packet list
 *
 * @sft_status [in]: SFT status associated with the flow
 * @ctx [in]: Worker context
 * @first_packet [in]: First packet in the fragmented packet list, returned by rte_sft_process_mbuf()
 */
static void
handle_fragmented_flow(struct rte_sft_flow_status *sft_status,
			struct worker_ctx *ctx,
			struct rte_mbuf *first_packet)
{
	int ret;
	struct rte_mbuf *drain_buff[BURST_SIZE];
	struct flow_info *flow;
	struct rte_sft_error error;
	struct rte_sft_mbuf_info mbuf_info = {0};
	struct rte_mbuf *packet = NULL;
	struct doca_dpi_parsing_info parsing_info = {0};
	int nb_packets_to_drain = sft_status->nb_ip_fragments, drained_packets;
	bool first_packet_enqueued = false;
	uint8_t packet_index;

	assert(first_packet != NULL);
	do {
		DOCA_DLOG_DBG("Draining %d fragmented packets, queue_id %d", nb_packets_to_drain, ctx->queue_id);
		ret = rte_sft_drain_fragment_mbuf(ctx->queue_id,
						SFT_ZONE,
						sft_status->ipfrag_ctx,
						nb_packets_to_drain,
						drain_buff,
						sft_status,
						&error);
		if (unlikely(ret != 0)) {
			DOCA_LOG_DBG("Failed to drain fragmented packets, error=%s", error.message);
			return;
		}

		drained_packets = sft_status->nb_ip_fragments;
		nb_packets_to_drain -= drained_packets;
		/* Program does not support fragments when they are the first ones in the flow */
		if (!sft_status->activated) {
			DOCA_LOG_DBG("No flow activated, droppisng packet");
			return;
		}
		if (!first_packet_enqueued) {
			/*
			 * Only after draining the frags, we get FID, so now we can enqueue first packet.
			 * First packet contains the L4 information and flow is determined.
			 */
			DOCA_DLOG_DBG("Enqueueing first packet");
			if (rte_sft_parse_mbuf(first_packet, &mbuf_info, NULL, &error) != 0) {
				DOCA_LOG_DBG("SFT parse MBUF failed, error=%s", error.message);
				return;
			}
			/* Verify that the l4 protocol is UDP or TCP */
			if (set_l4_parsing_info(&mbuf_info, &parsing_info) != DOCA_SUCCESS)
				return;
			flow = (struct flow_info *)rte_sft_flow_get_client_obj(ctx->queue_id,
									       sft_status->fid,
									       SFT_FLOW_INFO_CLIENT_ID,
									       &error);
			if (flow == NULL) {
				DOCA_LOG_ERR("SFT flow get client obj failed, error=%s", error.message);
				return;
			}
			enqueue_packet_to_dpi(first_packet, flow, sft_status, ctx);
			first_packet_enqueued = true;
		}
		DOCA_DLOG_DBG("Drained %d packets. Enqueueing them", drained_packets);

		for (packet_index = 0; packet_index < drained_packets; packet_index++) {
			packet = drain_buff[packet_index];
			enqueue_packet_to_dpi(packet, flow, sft_status, ctx);
		}
	} while (nb_packets_to_drain > 0);
}

/*
 * Handles OOO packets once they are received from SFT
 *
 * @sft_status [in]: SFT status associated with the flow
 * @flow [in]: Flow info associated with the flow
 * @ctx [in]: Worker context
 */
static void
handle_and_forward_ooo(struct rte_sft_flow_status *sft_status, struct flow_info *flow, struct worker_ctx *ctx)
{
	int drained_packets;
	int  packets_to_drain = sft_status->nb_in_order_mbufs;
	uint16_t packet_idx;
	struct rte_sft_error error;
	struct rte_mbuf *drain_buff[BURST_SIZE];
	struct rte_mbuf *packet = NULL;

	do {
		DOCA_DLOG_DBG("Draining %d OOO packets", packets_to_drain);
		drained_packets = rte_sft_drain_mbuf(ctx->queue_id,
						sft_status->fid,
						drain_buff,
						BURST_SIZE,
						sft_status->initiator,
						sft_status,
						&error);
		DOCA_DLOG_DBG("Drained %d packets", drained_packets);

		packets_to_drain -= drained_packets;
		if (drained_packets < 0) {
			DOCA_DLOG_DBG("Failed to drain packets, error=%s", error.message);
			return;
		}

		for (packet_idx = 0; packet_idx < drained_packets; packet_idx++) {
			packet = drain_buff[packet_idx];
			enqueue_packet_to_dpi(packet, flow, sft_status, ctx);
		}
	} while (packets_to_drain > 0);
}


uint16_t get_payload_length(struct rte_mbuf* packet) {
    return packet->data_len - sizeof(struct rte_ether_hdr);
}



void set_checksums(struct rte_mbuf *packet) {
    // Offload checksum calculation to the HW
    struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(packet, struct rte_ether_hdr *);
    struct rte_ipv4_hdr *ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
    struct rte_tcp_hdr *tcp_hdr = (struct rte_tcp_hdr *)(ip_hdr + 1);

	ip_hdr->hdr_checksum = 0;
	tcp_hdr->cksum = 0;	   
    packet->ol_flags |= RTE_MBUF_F_TX_IPV4 |  RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM;
}

void set_final(struct rte_mbuf *packet) {
    // Offload checksum calculation to the HW
    struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(packet, struct rte_ether_hdr *);
    struct rte_ipv4_hdr *ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
    struct rte_tcp_hdr *tcp_hdr = (struct rte_tcp_hdr *)(ip_hdr + 1);

	tcp_hdr->tcp_flags |= RTE_TCP_FIN_FLAG;

	ip_hdr->hdr_checksum = 0;
	tcp_hdr->cksum = 0;	   
    packet->ol_flags |= RTE_MBUF_F_TX_IPV4 |  RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM;
}

// Function to initialize the dynamic packet array
void init_packet_array(struct PacketArray *array) {
    array->packets = (struct rte_mbuf **)malloc(BURST_SIZE * sizeof(struct rte_mbuf *));
    if (array->packets == NULL) {
        // Handle memory allocation failure
        exit(1);
    }
    array->size = 0;
    array->capacity = BURST_SIZE;
}

// Function to add a packet reference to the dynamic packet array
void add_packet(struct PacketArray *array, struct rte_mbuf *packet) {
    if (array->size == array->capacity) {
        // Expand the array if needed
        array->capacity *= 2;
        array->packets = (struct rte_mbuf **)realloc(array->packets, array->capacity * sizeof(struct rte_mbuf *));
        if (array->packets == NULL) {
            // Handle memory allocation failure
            exit(1);
        }
    }
    array->packets[array->size++] = packet;
}

/*
 * The filtering function that processes data from packets
 */
unsigned char* process_data(const unsigned char *data_buffer, size_t data_size, size_t *filtered_size) {
    
	// Calculate the new size for the filtered buffer (half of data_size)
	// Arbitrary for now
    size_t new_size = data_size / 2;

    // Allocate memory for the filtered buffer
    unsigned char *filtered_buffer = (unsigned char *)malloc(new_size);
    if (filtered_buffer == NULL) {
        // Handle memory allocation failure
        return NULL;
    }

    // Copy the first half of the original data_buffer to the filtered_buffer
    memcpy(filtered_buffer, data_buffer, new_size);

    // Set the filtered_size to the new size
    *filtered_size = new_size;

    return filtered_buffer;
}


/*
 * The main function which polls burst of packets on the corresponding port on this lcore's queue
 * Each packet is processed by the SFT and then enqueued to the DPI engine
 *
 * @ctx [in]: Worker context
 */
static void
process_packet(struct worker_ctx *ctx)
{
	int ret = 0, packet_idx = 0;

	const uint8_t queue_id = ctx->queue_id;
	const uint16_t ingress_port = ctx->ingress_port;
	const uint16_t egress_port = ingress_port ^ 1;

   

	struct rte_sft_flow_status sft_status;
	struct rte_sft_error sft_error;

	struct rte_mbuf *buf_in[BURST_SIZE];
	struct rte_mbuf *packet;
	struct rte_mbuf *sft_packet;	/* Packet returned by SFT */

	struct flow_info *flow;
	const uint16_t nb_rx = rte_eth_rx_burst(ingress_port, queue_id, buf_in, BURST_SIZE);
	doca_error_t result;

	uint32_t payload_length;

	struct rte_sft_error error;
	struct rte_sft_mbuf_info mbuf_info;
	uint32_t payload_offset;

	const unsigned char *data;
	uint16_t header_len;
	struct rte_mbuf* new_packet = NULL;
	struct rte_mbuf* fin_packet = NULL;
	unsigned char* new_payload = "5\n";
	
	uint32_t o_seq = 0; //original sequence number of the last packet
	uint32_t o_len = 0; //original length
	//uint32_t diff_len = 0; //length diff between original and new
	uint32_t finalack = 0;
	bool tx = true;
	bool done = false;
	uint32_t tempack;
	uint32_t tempseq;
	

	struct rte_ether_hdr *eth_hdr1;
    struct rte_ipv4_hdr *ip_hdr1;
    struct rte_tcp_hdr *tcp_hdr1;
	

	ctx->processed_packets += nb_rx;

    int len;

	uint32_t tsval; 
    uint32_t tsecr; 
    
	/* Inspect each packet in the buffer */
	for (packet_idx = 0; packet_idx < nb_rx; packet_idx++) {
		DOCA_DLOG_DBG("================================ port = %d =============================================", ingress_port);

		packet = buf_in[packet_idx];

		/********************/
		eth_hdr1 = rte_pktmbuf_mtod(packet, struct rte_ether_hdr *);
    	ip_hdr1 = (struct rte_ipv4_hdr *)(eth_hdr1 + 1);
    	tcp_hdr1 = (struct rte_tcp_hdr *)(ip_hdr1 + 1);

		payload_offset = get_payload_offset(packet);
		bool src_flag = get_src(packet);

		uint16_t p_length = packet->pkt_len - payload_offset;	
         
        // Check if the dynamic buffer has enough space for the new data
        if (total_data_size + p_length - 1 > buffer_size) {
            // If not, resize the buffer using realloc
            size_t new_buffer_size = buffer_size * 2;  // Double the size, for example
            uint8_t *new_data_buffer = (uint8_t *)realloc(data_buffer, new_buffer_size);
            if (new_data_buffer == NULL) {
                fprintf(stderr, "Memory reallocation failed\n");
                free(data_buffer);
                return 1;
            }
            data_buffer = new_data_buffer;
            buffer_size = new_buffer_size;
        }


		//Catching the latest packet that contains which file to properly catch the timestamp
        //This packet will be used to send ACK from DPU to server
		if(ackcaught==false){
			if(ack_packet == NULL && !src_flag){
				if (tcp_hdr1->tcp_flags & RTE_TCP_ACK_FLAG && p_length > 0){
					//need to create new ack packet without timestamp in the mod_packet_pool
					ack_packet = rte_pktmbuf_copy(packet, mod_packet_pool, 0, payload_offset);
					ack_packet->pkt_len = payload_offset;
					ackcaught = true;
		    	}
		    }
		}


        //Catching payload data from server and buffer in the DPU
        if (src_flag && p_length > 0){

			// Change the name new to "start"?
			if(first == true){    
				//Copying first ACK,PUSH packet from server side
				data_packet = rte_pktmbuf_copy(packet, mod_packet_pool, 0,  packet->pkt_len);
				data_packet->pkt_len =  packet->pkt_len;
				first = false;
			}

        	// Copy payload data to the buffer
        	rte_memcpy(data_buffer + total_data_size, rte_pktmbuf_mtod_offset(packet, unsigned char*, payload_offset), p_length);
			total_data_size += p_length;

			add_packet(&packet_array, packet);

            
			print_tcp_header_info(packet);

			//Catch last packet  
			if (tcp_hdr1->tcp_flags & TCP_FIN_FLAG) {
				end = true;
				last_packet = rte_pktmbuf_copy(packet, mod_packet_pool, 0,  packet->pkt_len);
				last_packet->pkt_len =  packet->pkt_len;
			}

            
			//Need to check if it's actually needed or not
			//rte_pktmbuf_free(packet);

            //If we already have caught the ACK packet from client, send ACKs from DPU -> Server
			if(ack_packet!= NULL){  

				struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(ack_packet, struct rte_ether_hdr *);
    			struct rte_ipv4_hdr *ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
    			struct rte_tcp_hdr *tcp_hdr = (struct rte_tcp_hdr *)(ip_hdr + 1);

               

				tempack = rte_be_to_cpu_32(tcp_hdr1->sent_seq) + p_length;
				tempseq = rte_be_to_cpu_32(tcp_hdr1->recv_ack);

				update_pkt_header(ack_packet, ack_packet->pkt_len);

	            //TODO: check if this is actually needed or not
				uint16_t tmpwin = rte_be_to_cpu_16(tcp_hdr->rx_win);
                tcp_hdr->rx_win = rte_cpu_to_be_16(tmpwin);

				ack_packet->ol_flags |= RTE_MBUF_F_TX_IEEE1588_TMST;
				//Offload checksum calculation to the hardware
                ip_hdr->hdr_checksum = 0;
		        tcp_hdr->cksum = 0;	   
    			ack_packet->ol_flags |= RTE_MBUF_F_TX_IPV4 |  RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM;


			    if (end) {

				    tempack += 1; // Not sure why we have to add 1, but it works this way for the last packet
					tcp_hdr->sent_seq = rte_cpu_to_be_32(tempseq);
					tcp_hdr->recv_ack = rte_cpu_to_be_32(tempack);

                	// Add ACK and FIN flags
					tcp_hdr->tcp_flags &= RTE_TCP_ACK_FLAG;
                    tcp_hdr->tcp_flags |= RTE_TCP_ACK_FLAG | RTE_TCP_FIN_FLAG;
					
					new = true;

				}else{

					tcp_hdr->tcp_flags &= RTE_TCP_ACK_FLAG;
                    // Set the ACK flag
                    tcp_hdr->tcp_flags |= RTE_TCP_ACK_FLAG;

	                tcp_hdr->sent_seq = rte_cpu_to_be_32(tempseq);
				    tcp_hdr->recv_ack = rte_cpu_to_be_32(tempack);
				}

                TX_BUFFER_PKT(ack_packet, ctx);
				rte_eth_tx_buffer_flush(egress_port, queue_id, ctx->tx_buffer[egress_port]);


                //Send data packets from DPU to client
				if (new){
					
   					size_t filtered_size;

					unsigned char *filtered_buffer = processParquetData(data_buffer, total_data_size, &filtered_size);

                    //unsigned char *filtered_buffer = process_data(data_buffer, total_data_size, &filtered_size);
                    
    				if (filtered_buffer != NULL) {
        			
						//size_t packet_count = (filtered_size + PACKET_SIZE - 1) / PACKET_SIZE; // Calculate number of packets needed

                        size_t remaining_data = filtered_size;

                        size_t data_offset = 0;

                        for (size_t i = 0; i < packet_array.size; ++i) {

                        	struct rte_mbuf *new_packet = packet_array.packets[i];

							uint32_t p_offset = get_payload_offset(new_packet);
		        
				            size_t packet_capacity = new_packet->pkt_len - p_offset;

                            size_t chunk_size = packet_capacity < remaining_data ? packet_capacity : remaining_data;

                            unsigned char *chunk_data = data_buffer + data_offset;
                            
							unsigned char *payload = rte_pktmbuf_mtod_offset(new_packet, unsigned char *, p_offset);
                            rte_memcpy(payload, chunk_data, chunk_size);

                            remaining_data -= chunk_size;
                            data_offset += chunk_size;

                            if (remaining_data == 0) {
								set_final(new_packet);
								TX_BUFFER_PKT(new_packet, ctx);
            					break; // All data has been distributed among packets
        					}else{
								set_checksums(new_packet);
								TX_BUFFER_PKT(new_packet, ctx);
							}
    					}					
        				/*for (size_t i = 0; i < packet_count; ++i) {							
            			
						// Calculate start and end positions for the current packet
            				size_t start_pos = i * PACKET_SIZE;
            				size_t end_pos = (start_pos + PACKET_SIZE) < filtered_size ? (start_pos + PACKET_SIZE) : filtered_size;
            				size_t packet_data_length = end_pos - start_pos;

            				// Create a new mbuf packet for DPDK
            				struct rte_mbuf *packet = rte_pktmbuf_alloc(pkt_pool);
            				if (packet == NULL) {
                				// Handle allocation failure
                				free(filtered_buffer);
                				return 1;
            				}

		   					 // Copy data from filtered_buffer to the packet
            				rte_memcpy(rte_pktmbuf_mtod(packet, void *), filtered_buffer + start_pos, packet_data_length);
            				rte_pktmbuf_data_len(packet) = packet_data_length;
            				rte_pktmbuf_pkt_len(packet) = packet_data_length;

            				// Populate headers and set checksums as needed

            				// Send the packet using DPDK

            				// Free the packet
            				TX_BUFFER_PKT(packet, ctx);

        				}*/

        				// Filter the data buffer
        				free(filtered_buffer);
    				}
				    //write process_data function
					
					//Find num packets and send packets

                    //set_checksums(data_packet);
					//TX_BUFFER_PKT(data_packet, ctx);

					//set_checksums(last_packet);
					//TX_BUFFER_PKT(last_packet, ctx);
					
					rte_eth_tx_buffer_flush(egress_port, queue_id, ctx->tx_buffer[egress_port]);
					//After this need to re initialize everything to get new connection
					
				}

			}

            //do not send the packets that have data
			continue;
		}

        //Send all except we reach 
        //if(end == false || new == true){
		
		if(new == false){
		TX_BUFFER_PKT(packet, ctx);
		}
		//else free the packet?

/***********/
		

    	//struct rte_mbuf* new_packet = rte_pktmbuf_copy(packet, mod_packet_pool, 0, packet->pkt_len);

	
		header_len = payload_offset;

		o_len = packet->pkt_len;

        new_packet = packet;

		done = false;

        /*Check whether it is the server side sending data*/
		if (src_flag && done){	

			eth_hdr1 = rte_pktmbuf_mtod(packet, struct rte_ether_hdr *);
    	    ip_hdr1 = (struct rte_ipv4_hdr *)(eth_hdr1 + 1);
    	    tcp_hdr1 = (struct rte_tcp_hdr *)(ip_hdr1 + 1);

			if (p_length > 0){


				/*new_packet = rte_pktmbuf_copy(packet, mod_packet_pool, 0, header_len);
				if (new_packet == NULL) {
					printf("Alloc failed");
        			return;
    			}else{
			    	packet = new_packet;
				}*/


				new_packet->pkt_len = strlen(new_payload) + header_len;
				new_packet->data_len = strlen(new_payload) + header_len;

				diff_len = o_len - new_packet->pkt_len;

            
    			unsigned char* src = rte_pktmbuf_mtod_offset(new_packet, unsigned char*, payload_offset);  // Cast sft_packet to unsigned char* source pointer
    			rte_memcpy(src, new_payload, strlen(new_payload));
	
				unsigned char* pdata = rte_pktmbuf_mtod_offset(new_packet, unsigned char*, payload_offset);

				uint16_t p_length = new_packet->pkt_len - payload_offset;


    			/*for (int i = 0; i < p_length; i++) {
       				printf("%c", pdata[i]);
    			}
    			printf("\n");

				printf("Data len %d \n", new_packet->data_len);
				printf("Pkt data len %d \n", new_packet->pkt_len);*/

	
    			struct rte_ether_hdr *eth_hdr = rte_pktmbuf_mtod(new_packet, struct rte_ether_hdr *);
    			struct rte_ipv4_hdr *ip_hdr = (struct rte_ipv4_hdr *)(eth_hdr + 1);
    			struct rte_tcp_hdr *tcp_hdr = (struct rte_tcp_hdr *)(ip_hdr + 1);

				/*This is not needed for non modified packet size*/
				update_pkt_header(new_packet, new_packet->pkt_len);


				new_packet->l2_len = sizeof(*eth_hdr);
    			new_packet->l3_len = sizeof(*ip_hdr);
    			ip_hdr->hdr_checksum = 0;
				tcp_hdr->cksum = 0;
    			new_packet->ol_flags |= RTE_MBUF_F_TX_IPV4 |  RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM;

			

				if (!first){
					tcp_hdr->sent_seq = rte_cpu_to_be_32(finalseq);
				}		
			
				//print_tcp_header_info(new_packet);
      
		
		    	//printf("Seq of data %u\n", finalseq );
	
                finalseq = rte_be_to_cpu_32(tcp_hdr->sent_seq) + p_length ;
				first = false;

			}else{
				
                /*Server send FIN packet to notify the end of packets
				the FIN packets is sent at the same time as the data packet
				Need to catch it and change it*/
		        if ((tcp_hdr1->tcp_flags & TCP_FIN_FLAG)) { 


                    o_seq = rte_be_to_cpu_32(tcp_hdr1->sent_seq) - diff_len;
					tcp_hdr1->sent_seq = rte_cpu_to_be_32(o_seq);

    			    ip_hdr1->hdr_checksum = 0;
				    tcp_hdr1->cksum = 0;	   
    			    packet->ol_flags |= RTE_MBUF_F_TX_IPV4 |  RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM;

				}
			}		
		}


		
/*End*/
		DOCA_DLOG_DBG("================================================================================");
		/* Add additional new lines for output readability */
		DOCA_DLOG_DBG("\n");
	}


	//dequeue_from_dpi(ctx);
	/* Flush ready to send packets */
	rte_eth_tx_buffer_flush(egress_port, queue_id, ctx->tx_buffer[egress_port]);
	
	clear_aged_flows(ctx);
}

/*
 * Cleanup worker context
 *
 * @ctx [in]: Worker context
 */
static void
dpi_worker_cleanup(struct worker_ctx *ctx)
{
	if (ctx == NULL)
		return;
	if (ctx->workq != NULL) {
		doca_ctx_workq_rm(doca_dpi_as_ctx(ctx->attr.dpi_ctx->dpi), ctx->workq);
		doca_workq_destroy(ctx->workq);
		ctx->workq = NULL;
	}
	if (ctx->doca_dpdk_pool != NULL) {
		doca_dpdk_mempool_destroy(ctx->doca_dpdk_pool);
		ctx->doca_dpdk_pool = NULL;
	}
	if (ctx->buf_inventory != NULL) {
		doca_buf_inventory_destroy(ctx->buf_inventory);
		ctx->buf_inventory = NULL;
	}
	if (ctx->meta_mempool != NULL) {
		rte_mempool_free(ctx->meta_mempool);
		ctx->meta_mempool = NULL;
	}
	rte_free(ctx);
}

/*
 * The lcore main. This is the main thread that does the work, reading from
 * an input port and writing to an output port.
 *
 * @worker [in]: Worker context
 */
static void
dpi_worker(void *worker)
{
	uint8_t nb_ports = rte_eth_dev_count_avail();
	uint8_t port;
	struct worker_ctx *ctx = (struct worker_ctx *)worker;
	const uint16_t buff_size = BURST_SIZE;

	ackcaught=false;
	ack_packet = NULL;
    ackpacket = NULL;
	data_packet = NULL;
	last_packet = NULL;
	end = false;
	init_packet_array(&packet_array);


    /*Data buffer to contain payload data from the server*/
    data_buffer = (unsigned char *)malloc(INITIAL_BUFFER_SIZE);
    if (data_buffer == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
	total_data_size = 0;
    buffer_size = INITIAL_BUFFER_SIZE;
	/*****/

    

	ctx->dropped_packets = 0;
	for (port = 0; port < nb_ports; port++) {
		ctx->tx_buffer[port] = rte_zmalloc_socket(NULL, RTE_ETH_TX_BUFFER_SIZE(buff_size), 0, rte_socket_id());
		if (rte_eth_tx_buffer_init(ctx->tx_buffer[port], buff_size) != 0) {
			force_quit = true;
			DOCA_LOG_ERR("Failed to init TX buffer");
			goto thread_out;
		}
		rte_eth_tx_buffer_set_err_callback(ctx->tx_buffer[port], rte_eth_tx_buffer_count_callback, &ctx->dropped_packets);
	}

	DOCA_DLOG_DBG("Core %u is forwarding packets", rte_lcore_id());
	/* Run until the application is quit or killed */
	while (!force_quit) {
		for (port = 0; port < nb_ports; port++) {
			ctx->ingress_port = port;
			process_packet(ctx);
		}
	}

    free(packet_array.packets);
  // Open a file for writing
   FILE *output_file = fopen("output.txt", "wb"); // "wb" for binary mode

    if (output_file == NULL) {
        fprintf(stderr, "Failed to open output file\n");
        free(data_buffer);
        return 1;
    }

    // Write each element of data_buffer to the file
    for (size_t i = 0; i < total_data_size; i++) {
        fwrite(&data_buffer[i], sizeof(unsigned char), 1, output_file);
    }


   

    // Close the file
    fclose(output_file);
	free(data_buffer);

thread_out:
	for (port = 0; port < nb_ports; port++)
		if (ctx->tx_buffer[port] != NULL) {
			rte_eth_tx_buffer_flush(port, 0, ctx->tx_buffer[port]);
			rte_free(ctx->tx_buffer[port]);
		}
	pthread_mutex_lock(&log_lock);
	DOCA_LOG_INFO("Core %u has processed %lu packets, dropped %lu packets", rte_lcore_id(),
									ctx->processed_packets, ctx->dropped_packets);
	pthread_mutex_unlock(&log_lock);
	dpi_worker_cleanup(ctx);
}

void
dpi_worker_lcores_stop(struct doca_dpi_worker_ctx *dpi_ctx)
{
	struct doca_dpi_stat_info doca_stat = {0};

	force_quit = true;
	rte_eal_mp_wait_lcore();
	/* Print DPI statistics */
	doca_dpi_get_stats(dpi_ctx->dpi, true, &doca_stat);
	DOCA_LOG_INFO("------------- DPI STATISTICS --------------");
	DOCA_LOG_INFO("Packets scanned:%d", doca_stat.nb_scanned_pkts);
	DOCA_LOG_INFO("Matched signatures:%d", doca_stat.nb_matches);
	DOCA_LOG_INFO("TCP matches:%d", doca_stat.nb_tcp_based);
	DOCA_LOG_INFO("UDP matches:%d", doca_stat.nb_udp_based);
	DOCA_LOG_INFO("HTTP matches:%d", doca_stat.nb_http_parser_based);
	DOCA_LOG_INFO("SSL matches:%d", doca_stat.nb_ssl_parser_based);
	DOCA_LOG_INFO("Miscellaneous L4:%d, L7:%d", doca_stat.nb_other_l4, doca_stat.nb_other_l7);
}

void
printf_signature(struct doca_dpi_worker_ctx *dpi_ctx, uint32_t sig_id, uint32_t fid, bool blocked)
{
	doca_error_t result;
	struct doca_dpi_sig_data sig_data;

	result = doca_dpi_get_signature(dpi_ctx->dpi, sig_id, &sig_data);
	if (likely(result == DOCA_SUCCESS))
		DOCA_LOG_INFO("SIG ID: %u, APP Name: %s, SFT_FID: %u, Blocked: %u", sig_id, sig_data.name, fid, blocked);
	else
		DOCA_LOG_ERR("Failed to get signatures: %s", doca_get_error_string(result));
}

/*
 * Free callback - free doca_buf allocated pointer
 *
 * @addr [in]: Memory range pointer
 * @len [in]: Memory range length
 * @opaque [in]: An opaque pointer passed to iterator
 */
void
free_cb(void *addr, size_t len, void *opaque)
{
	(void)len;
	(void)opaque;

	free(addr);
}

doca_error_t
dpi_worker_lcores_run(int nb_queues, struct dpi_worker_attr attr)
{
	int lcore = 0;
	uint16_t lcore_index = 0;
	struct worker_ctx *ctx = NULL; /* To be freed by the worker */
	char name[MEMPOOL_NAME_MAX_LEN];
	doca_error_t result;
	uint8_t nb_ports = rte_eth_dev_count_avail();



	if (nb_ports != SFT_PORTS_NUM) {
		DOCA_LOG_ERR("Application only function with [%d] ports, but num_of_ports = [%d]",
				SFT_PORTS_NUM, nb_ports);
		return DOCA_ERROR_INVALID_VALUE;
	}


	/* Make sure the value is initialized */
	force_quit = false;


	/*Modified*/
	mod_packet_pool = attr.dpdk_config->mbuf_pool;

	/* Main thread is reserved */
	RTE_LCORE_FOREACH_WORKER(lcore) {
		DOCA_DLOG_DBG("Creating worker on core %u", lcore);
		ctx = (struct worker_ctx *)rte_zmalloc(NULL, sizeof(struct worker_ctx), 0);
		if (ctx == NULL) {
			DOCA_LOG_ERR("Failed to allocate memory for worker context on core %d", lcore);
			return DOCA_ERROR_NO_MEMORY;
		}
		ctx->queue_id = lcore_index;
		ctx->attr = attr;

		/* To prepare the job result buffer for the DPI engine */
		sprintf(name, "meta_mempool_%d", ctx->queue_id);
		ctx->meta_mempool = rte_mempool_create(name,
							NUM_MBUFS,
							sizeof(struct dpi_job_meta_data),
							0, 0,
							NULL, NULL,
							NULL, NULL,
							rte_socket_id(),
							RTE_MEMPOOL_F_SC_GET | RTE_MEMPOOL_F_SP_PUT);
		if (ctx->meta_mempool == NULL) {
			DOCA_LOG_ERR("Failure to allocate results mempool!");
			dpi_worker_cleanup(ctx);
			return DOCA_ERROR_NO_MEMORY;
		}
		
	

		/* DOCA buf_inventory create and start for DPDK bridge */
		result = doca_buf_inventory_create(NULL, NUM_MBUFS, DOCA_BUF_EXTENSION_NONE, &ctx->buf_inventory);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Buf_inventory create failure, %s", doca_get_error_string(result));
			dpi_worker_cleanup(ctx);
			return result;
		}
		result = doca_buf_inventory_start(ctx->buf_inventory);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Buf_inventory start failure, %s", doca_get_error_string(result));
			dpi_worker_cleanup(ctx);
			return result;
		}

		/*
		 * DOCA DPDK bridge create and start.
		 * So the rte_mbuf can be converted into DOCA buf.
		 */
		result = doca_dpdk_mempool_create(attr.dpdk_config->mbuf_pool, &ctx->doca_dpdk_pool);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("DPDK bridge_mempool create failure, %s", doca_get_error_string(result));
			dpi_worker_cleanup(ctx);
			return result;
		}
		result = doca_dpdk_mempool_dev_add(ctx->doca_dpdk_pool, attr.dpi_ctx->dev);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("DOCA DPDK mempool dev add failure, %s", doca_get_error_string(result));
			dpi_worker_cleanup(ctx);
			return result;
		}
		result = doca_dpdk_mempool_start(ctx->doca_dpdk_pool);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("DOCA DPDK mempool start failure, %s", doca_get_error_string(result));
			dpi_worker_cleanup(ctx);
			return result;
		}



/*compress job start*/
/*Compress*/
		/*
			struct doca_mmap *dst_mmap;		/* doca mmap for destination buffer */
/*	struct doca_buf_inventory *buf_inv;	/* doca buffer inventory */
	
/*	char *dst_buffer;
	struct doca_buf *dst_doca_buf;
	dst_buffer = calloc(1, 1024);
	//strcpy(dst_buffer, "bye");
	//state.ctx = doca_compress_as_ctx(compress);
	char pci_address[PCI_ADDR_LEN];
	struct doca_pci_bdf pcie_dev;

	struct program_core_objects state = {0};
	uint32_t workq_depth = 1;		/* The sample will run 1 compress job */
	/*uint32_t max_bufs = 2;		/* The sample will use 2 doca buffers */
	//void* data;


	/*if (dst_buffer == NULL) {
		printf("Failed to allocate memory\n");
	}
	/******Mod***/
	
	//strcpy(pci_address, "03:00.0");
	//result = doca_pci_bdf_from_string(pci_address, &pcie_dev);

    //esult = open_doca_device_with_pci(pcie_dev, &compress_jobs_compress_is_supported, &state.dev);

   
	//result = start_context(state, workq_depth);
    /*printf("CreATEDE CORE OBJETCS");
	if (result != DOCA_SUCCESS) {
		printf("Unable to initialize: %s\n", doca_get_error_string(result));
		return result;
	}

	result =  doca_mmap_create(NULL, &state.dst_mmap);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create destination mmap: %s\n", doca_get_error_string(result));
		return result;
	}
	result = doca_mmap_dev_add(state.dst_mmap, state.dev);
	if (result != DOCA_SUCCESS) {
		printf("Unable to add device to destination mmap: %s\n", doca_get_error_string(result));
		doca_mmap_destroy(state.dst_mmap);
		state.dst_mmap = NULL;
		return result;
	}

	result = doca_buf_inventory_create(NULL, 1, DOCA_BUF_EXTENSION_NONE, &state.buf_inv);
	if (result != DOCA_SUCCESS) {
		printf("Unable to create buffer inventory: %s\n", doca_get_error_string(result));
		return result;
	}

	result = doca_buf_inventory_start(state.buf_inv);
	if (result != DOCA_SUCCESS) {
		printf("Unable to start buffer inventory: %s\n", doca_get_error_string(result));
		return result;
	}


    result = doca_mmap_set_memrange(state.dst_mmap, dst_buffer, 1024);
	if (result != DOCA_SUCCESS) {
		printf("Failed to allocate memrange\n");
		free(dst_buffer);
		return result;
	}
	result = doca_mmap_set_free_cb(state.dst_mmap, &free_cb, NULL);
	if (result != DOCA_SUCCESS) {
		printf("Failed to allocate cb\n");
		free(dst_buffer);
		return result;
	}
	result = doca_mmap_start(state.dst_mmap);
	if (result != DOCA_SUCCESS) {
		printf("Failed to start\n");
		free(dst_buffer);
		return result;
	}

	
	result =  doca_buf_inventory_buf_by_addr(state.buf_inv, state.dst_mmap, dst_buffer,1024, &dst_doca_buf);
	if (result != DOCA_SUCCESS) {
		printf("Unable to acquire DOCA buffer representing destination buffer: %s", doca_get_error_string(result));
		return result;
	}

//free(dst_buffer);

/******************/



		/* Workq create and add */
		result = doca_workq_create(NUM_MBUFS, &ctx->workq);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Unable to create DOCA workq, err = [%s]",
				doca_get_error_string(result));
			rte_free(ctx);
			return result;
		}
		result = doca_ctx_workq_add(doca_dpi_as_ctx(attr.dpi_ctx->dpi), ctx->workq);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Unable to register workq with doca_dpi, err = [%s]",
					doca_get_error_string(result));
			dpi_worker_cleanup(ctx);
			return result;
		}

		if (rte_eal_remote_launch((void *)dpi_worker, (void *)ctx, lcore) != 0) {
			DOCA_LOG_ERR("Failed to launch DPI worker on core %d", lcore);
			dpi_worker_cleanup(ctx);
			return DOCA_ERROR_DRIVER;
		}
		lcore_index++;
	}

	if (lcore_index != nb_queues) {
		DOCA_LOG_ERR("%d cores are used as DPI workers, but %d queues are configured", lcore_index, nb_queues);
		return DOCA_ERROR_INVALID_VALUE;
	}
	DOCA_LOG_INFO("%d cores are used as DPI workers", lcore_index);
	return DOCA_SUCCESS;
}