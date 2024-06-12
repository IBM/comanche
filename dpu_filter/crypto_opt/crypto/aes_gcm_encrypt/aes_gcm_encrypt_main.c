#include <stdlib.h>
#include <string.h>

#include <doca_argp.h>
#include <doca_aes_gcm.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <sys/mman.h>
#include <utils.h>

#include "aes_gcm_common.h"

DOCA_LOG_REGISTER(AES_GCM_ENCRYPT::MAIN);

/* Sample's Logic */
doca_error_t aes_gcm_encrypt(struct aes_gcm_cfg *cfg, char *file_data, size_t file_size);

/*
 * Sample main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
int
main(int argc, char **argv)
{
	doca_error_t result;
	struct aes_gcm_cfg aes_gcm_cfg;
	char *file_data = NULL;
	size_t file_size;
	struct doca_log_backend *sdk_log;
	int exit_status = EXIT_FAILURE;
	clock_t start_time, end_time;  // Timing variables
	struct timeval start, end;

	

	/* Register a logger backend */
	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		goto sample_exit;

	/* Register a logger backend for internal SDK errors and warnings */
	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		goto sample_exit;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		goto sample_exit;

	DOCA_LOG_INFO("Starting the sample");

	init_aes_gcm_params(&aes_gcm_cfg);

	result = doca_argp_init("doca_aes_gcm_encrypt", &aes_gcm_cfg);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		goto sample_exit;
	}

	result = register_aes_gcm_params();
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to register ARGP params: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	result = doca_argp_start(argc, argv);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to parse sample input: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}

	

	result = read_file(aes_gcm_cfg.file_path, &file_data, &file_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to read file: %s", doca_error_get_descr(result));
		goto argp_cleanup;
	}


    /*printf("Pin all the pages in memory.\n");
    if(mlockall(MCL_CURRENT | MCL_FUTURE) < 0) {
        perror("mlockall");
        return 1;
    }*/
	
	gettimeofday(&start, NULL);

	result = aes_gcm_encrypt(&aes_gcm_cfg, file_data, file_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("aes_gcm_encrypt() encountered an error: %s", doca_error_get_descr(result));
		goto data_file_cleanup;
	}


	gettimeofday(&end, NULL);
    double time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
    printf("Time taken: %.6f seconds\n", time_taken);


    /*printf("Finished, clean up.\n");
    munlockall();*/
	exit_status = EXIT_SUCCESS;

data_file_cleanup:
	if (file_data != NULL)
		free(file_data);
argp_cleanup:
	doca_argp_destroy();
sample_exit:
	if (exit_status == EXIT_SUCCESS)
		DOCA_LOG_INFO("Sample finished successfully");
	else
		DOCA_LOG_INFO("Sample finished with errors");
	return exit_status;
}
