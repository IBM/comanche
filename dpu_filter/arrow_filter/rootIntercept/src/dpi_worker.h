#ifndef COMMON_DPI_WORKER_H_
#define COMMON_DPI_WORKER_H_

#include <doca_dev.h>
#include <doca_dpi.h>

#include "telemetry.h"

#ifdef __cplusplus
extern "C" {
#endif

extern volatile bool force_quit; /* Used to communicate between the DPDK workers and the main thread */

#define MBUFS_WATERMARK (0.8)      /* limit num of mbuf which dpi library can use to X percentage of mbuf per core */
#define MAX_MBUFS_FOR_FLOW (10)    /* assuming this is the max number of packets each flow can hold - 
				    /* this will help calculate max num of flows per workq (num of buf/ max bufs per flow)
				    */

struct doca_dpi_worker_ctx {
	struct doca_dev *dev;				/* The DOCA DPI device */
	struct doca_dpi *dpi;				/* The DOCA DPI instance */
};

/*
 * The wrapper of whether doca_dpi is supported.
 *
 * @devinfo [in]: The device info.
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpi_job_is_supported(struct doca_devinfo *devinfo);

/*
 * Init doca_dpi internal objects
 *
 * @dpi_ctx [in]: DOCA DPI worker containing all the necessary objects.
 * @max_sig_match_len [in]: The maximum signature match length.
 * @per_workq_packet_pool_size [in]: The maximum inflight packets per queue.
 * @dev [in]: The doca_device opened with pci_address.
 * @sig_file_path [in]: The signature file path to be used for programming regex engine.
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t doca_dpi_worker_ctx_create(struct doca_dpi_worker_ctx **dpi_ctx,
					uint16_t max_sig_match_len,
					uint32_t per_workq_packet_pool_size,
					struct doca_dev *dev,
					const char *sig_file_path);

/*
 * Destroy doca_dpi internal objects
 *
 * @dpi_ctx [in]: DOCA DPI worker containing all the necessary objects.
 */
void doca_dpi_worker_ctx_destroy(struct doca_dpi_worker_ctx *dpi_ctx);

/* DPI worker action to take when a match is found */
enum dpi_worker_action {
	DPI_WORKER_ALLOW,	/* Allow the flow to pass and offload to HW */
	DPI_WORKER_DROP,	/* Drop the flow and offload to HW */
	DPI_WORKER_RSS_FLOW	/* Allow the flow to pass without offloading to HW */
};

/* To contain packet references of data packets*/
/*struct PacketArray {
    struct rte_mbuf **packets;
    size_t size;
    size_t capacity;
};*/

/* Callback function to be called when DPI engine matches a packet */
typedef int (*dpi_on_match_t)(const struct doca_dpi_result *result,
				uint32_t fid,
				void *user_data,
				enum dpi_worker_action *dpi_action);

/* Callback function to be called to send netflow records */
typedef void (*send_netflow_record_t)(const struct doca_telemetry_netflow_record *record);

/* DPI worker attributes */
struct dpi_worker_attr {
	dpi_on_match_t			dpi_on_match;		/* Will be called on DPI match */
	send_netflow_record_t		send_netflow_record;	/* Will be called when netflow record is ready to be sent */
	void				*user_data;		/* User data passed to dpi_on_match */
	uint64_t			max_dpi_depth;		/* Max DPI depth search limit, use 0 for unlimited depth */
	struct doca_dpi_worker_ctx	*dpi_ctx;		/* DOCA DPI context, passed to all workers */
	struct application_dpdk_config	*dpdk_config;		/* DPDK configuration */
};

/*
 * Prints DPI signature status
 *
 * @dpi_ctx [in]: DOCA DPI context
 * @sig_id [in]: DPI signature ID
 * @fid [in]: Flow ID
 * @blocked [in]: 1 if signature is blocked and 0 otherwise
 */
void printf_signature(struct doca_dpi_worker_ctx *dpi_ctx, uint32_t sig_id, uint32_t fid, bool blocked);

/*
 * This is the main worker calling function, each queue represents a core
 *
 * @available_cores [in]: Number of available cores
 * @attr [in]: DPI worker attributes
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t dpi_worker_lcores_run(int available_cores, struct dpi_worker_attr attr);

/*
 * Stops lcores and wait until all lcores are stopped.
 *
 * @dpi_ctx [in]: DPI context
 */
void dpi_worker_lcores_stop(struct doca_dpi_worker_ctx *dpi_ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* COMMON_DPI_WORKER_H_ */