#ifndef IPS_CORE_H_
#define IPS_CORE_H_

#include <doca_dpi.h>

#include <pi_worker.h>
#include <offload_rules.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_FILE_NAME 255			/* Maximal length of file path */
#define NETFLOW_QUEUE_SIZE 1024			/* Netflow queue size */
#define USER_PCI_ADDR_LEN 7			/* User PCI address string length */
#define PCI_ADDR_LEN (USER_PCI_ADDR_LEN + 1)

extern struct doca_dpi_state *dpi_state;	/* DPI context */

struct ips_config {
	struct application_dpdk_config *dpdk_config;	/* DPDK configuration */
	char cdo_filename[MAX_FILE_NAME];		/* Path to CDO file */
	char csv_filename[MAX_FILE_NAME];		/* Path to CSV file */
	bool create_csv;				/* CSV output flag */
	bool print_on_match;				/* Print on match flag */
	bool enable_fragmentation;			/* Enable fragmentation */
	int netflow_source_id;				/* Netflow source id */
	char pci_address[PCI_ADDR_LEN];			/* PCI address */
};

/*
 * IPS initialization function.
 * Initializes the IPS application and creates the DPI context.
 *
 * @app_dpdk_config [in]: application DPDK configuration values
 * @ips_config [in]: IPS configuration
 * @dpi_worker [in]: DPI worker attributes
 * @return: 0 on success and negative value otherwise
 */
int ips_init(const struct application_dpdk_config *app_dpdk_config, struct ips_config *ips_config,
		struct dpi_worker_attr *dpi_worker);

/*
 * IPS destroy
 *
 * @app_dpdk_config [in]: application DPDK configuration values
 * @ips [in]: application configuration structure
 */
void ips_destroy(struct application_dpdk_config *app_dpdk_config, struct ips_config *ips);

/*
 * Register the command line parameters for the IPS application
 *
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
doca_error_t register_ips_params(void);

/*
 * Callback function to handle TERM and INT signals
 *
 * @signum [in]: signal number
 */
void signal_handler(int signum);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* IPS_CORE_H_ */
