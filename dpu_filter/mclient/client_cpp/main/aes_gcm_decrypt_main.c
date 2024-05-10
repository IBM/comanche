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

DOCA_LOG_REGISTER(AES_GCM_DECRYPT::MAIN);

/* Sample's Logic */
uint8_t* aes_gcm_decrypt(struct aes_gcm_cfg *cfg, char *file_data, size_t file_size, size_t* output_size);

/*
 * Sample main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
uint8_t* decrypt_buffer(char* file_data, size_t file_size, size_t* output_size) 
{
	doca_error_t result;
	struct aes_gcm_cfg aes_gcm_cfg;
	struct doca_log_backend *sdk_log;
	int exit_status = EXIT_FAILURE;
	struct timeval start, end;


	/* Register a logger backend */
	/*result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		return NULL;

	/* Register a logger backend for internal SDK errors and warnings */
	/*result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		return NULL;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		return NULL;

	DOCA_LOG_INFO("Starting the sample");*/

	init_aes_gcm_params(&aes_gcm_cfg);  //trivial

	result = doca_argp_init("doca_aes_gcm_decrypt", &aes_gcm_cfg);  //trivial
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		return NULL;
	}

    


	gettimeofday(&start, NULL);
	uint8_t *output_data = aes_gcm_decrypt(&aes_gcm_cfg, file_data, file_size, output_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("aes_gcm_decrypt() encountered an error: %s", doca_error_get_descr(result));
		return NULL;
	}

	//printf("Output size: %u ms\n", *output_size);

	gettimeofday(&end, NULL);
    double time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
    //printf("Decryption time taken: %.6f seconds\n", time_taken);


	//DOCA_LOG_INFO("Decryption finished successfully");
	return output_data;

}
