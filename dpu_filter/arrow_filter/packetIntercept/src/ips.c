#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

#include <rte_sft.h>
#include <rte_malloc.h>
#include <rte_ring.h>

#include <doca_argp.h>
#include <doca_dpi.h>
#include <doca_log.h>

#include <dpdk_utils.h>
#include <utils.h>
#include <sig_db.h>

#include "ips_core.h"
#include "process_data.h" // Include your header file

DOCA_LOG_REGISTER(IPS);

/*
 * IPS application main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int
main(int argc, char *argv[])
{
	doca_error_t result;
	int ret;
	int exit_status = EXIT_SUCCESS;
	struct dpi_worker_attr dpi_worker_attr = {0};
	struct application_dpdk_config dpdk_config = {
		.port_config.nb_ports = 2,
		.port_config.nb_queues = 2,
		.port_config.nb_hairpin_q = 4,
		.port_config.enable_mbuf_metadata = 1,
		.sft_config = {
			.enable = true,
			.enable_ct = true,
			.enable_state_hairpin = false,
			.enable_state_drop = true
		},
		.reserve_main_thread = true,
	};
	struct ips_config ips_config = {.dpdk_config = &dpdk_config};
	struct doca_logger_backend *stdout_logger = NULL;


    //const char* parquet_filename = "data.parquet";
    //int cpp_result = processParquetFile(parquet_filename);

	/* Create a logger backend that prints to the standard output */
	result = doca_log_create_file_backend(stdout, &stdout_logger);
	if (result != DOCA_SUCCESS)
		return EXIT_FAILURE;

	/* Parse cmdline/json arguments */
	result = doca_argp_init("doca_ips", &ips_config);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_get_error_string(result));
		return EXIT_FAILURE;
	}
	doca_argp_set_dpdk_program(dpdk_init);
	result = register_ips_params();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register application params: %s", doca_get_error_string(result));
		doca_argp_destroy();
		return EXIT_FAILURE;
	}
	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse application input: %s", doca_get_error_string(result));
		doca_argp_destroy();
		return EXIT_FAILURE;
	}

	/* init DPDK cores and SFT */
	result = dpdk_queues_and_ports_init(&dpdk_config);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to update application ports and queues: %s", doca_get_error_string(result));
		exit_status = EXIT_FAILURE;
		goto dpdk_destroy;
	}

	ret = ips_init(&dpdk_config, &ips_config, &dpi_worker_attr);
	if (ret < 0) {
		exit_status = EXIT_FAILURE;
		goto dpdk_cleanup;
	}

	/* Start the DPI processing */
	result = dpi_worker_lcores_run(dpdk_config.port_config.nb_queues, dpi_worker_attr);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start DPI worker: %s", doca_get_error_string(result));
		exit_status = EXIT_FAILURE;
		goto ips_cleanup;
	}

	signal(SIGINT, signal_handler);
	signal(SIGTERM, signal_handler);

	/* The main thread */
	while (!force_quit) {
		sleep(1);
		if (ips_config.create_csv) {
			result = sig_database_write_to_csv(ips_config.csv_filename);
			if (result != DOCA_SUCCESS) {
				DOCA_LOG_ERR("CSV file access failed");
				exit_status = EXIT_FAILURE;
				goto ips_cleanup;
			}
		}
		if (ips_config.netflow_source_id && send_netflow_record() != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Unexpected Netflow failure");
			exit_status = EXIT_FAILURE;
			goto ips_cleanup;
		}
	}

ips_cleanup:
	/* End of application flow */
	ips_destroy(&dpdk_config, &ips_config);

dpdk_cleanup:
	/* DPDK cleanup */
	dpdk_queues_and_ports_fini(&dpdk_config);
dpdk_destroy:
	dpdk_fini();

	/* ARGP cleanup */
	doca_argp_destroy();

	return exit_status;
}
