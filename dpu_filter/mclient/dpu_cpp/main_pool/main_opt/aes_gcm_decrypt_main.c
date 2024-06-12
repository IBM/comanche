#include <stdlib.h>
#include <string.h>

#include <doca_argp.h>
#include <doca_aes_gcm.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <sys/mman.h>
#include <doca_ctx.h>
#include <utils.h>
#include "common.h"

#include "aes_gcm_common.h"

#define MAX_BUFFER_SIZE 1048576//2097152  // Define the maximum buffer size, 2MB

struct aes_gcm_resources resources = {0};
struct aes_gcm_cfg aes_gcm_cfg;

DOCA_LOG_REGISTER(AES_GCM_DECRYPT::MAIN);

/* Sample's Logic */
uint8_t* aes_gcm_decrypt(struct aes_gcm_cfg *cfg, char *file_data, size_t file_size, size_t* output_size, struct aes_gcm_resources *resources);


void init_crypto_resources(){

	doca_error_t result = DOCA_SUCCESS;
    struct doca_log_backend *sdk_log;

	/* Register a logger backend */
	result = doca_log_backend_create_standard();
	if (result != DOCA_SUCCESS)
		return NULL;

	/* Register a logger backend for internal SDK errors and warnings */
	result = doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
	if (result != DOCA_SUCCESS)
		return NULL;
	result = doca_log_backend_set_sdk_level(sdk_log, DOCA_LOG_LEVEL_WARNING);
	if (result != DOCA_SUCCESS)
		return NULL;

	init_aes_gcm_params(&aes_gcm_cfg);  //trivial

	result = doca_argp_init("doca_aes_gcm_decrypt", &aes_gcm_cfg);  //trivial
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to init ARGP resources: %s", doca_error_get_descr(result));
		return NULL;
	}
	resources.mode = AES_GCM_MODE_DECRYPT;
	result = allocate_aes_gcm_resources(aes_gcm_cfg.pci_address, 2, &resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to allocate AES-GCM resources: %s", doca_error_get_descr(result));
	}

		/* Start AES-GCM context */
	result = doca_ctx_start(resources.state->ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start context: %s", doca_error_get_descr(result));
	}

}

void destroy_crypto_resources(){

	doca_error_t tmp_result = DOCA_SUCCESS;
		
	doca_ctx_stop(resources.state->ctx); //this alone takes 323 msec


	tmp_result = destroy_aes_gcm_resources(&resources);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy AES-GCM resources: %s", doca_error_get_descr(tmp_result));
	}
}

/*
 * Sample main function
 *
 * @argc [in]: command line arguments size
 * @argv [in]: array of command line arguments
 * @return: EXIT_SUCCESS on success and EXIT_FAILURE otherwise
 */
uint8_t* decrypt_buffer(char* file_data, size_t file_size, size_t* output_size) 
{
	
	
	int exit_status = EXIT_FAILURE;
	struct timeval start, end;
	doca_error_t result;
	
    size_t buffer_size = 0;

    struct program_core_objects *state = NULL;
	state = resources.state;

	DOCA_LOG_INFO("Start sample");


    result = create_core_objects(state, 2);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create DOCA core objects: %s", doca_error_get_descr(result));
	}


	    //Max size that the crypto engine supports
	/*buffer_size = MAX_BUFFER_SIZE;

	// Calculate total buffer size
    size_t num_chunks = (file_size + buffer_size - 1) / buffer_size;
	//DOCA_LOG_INFO("Num chunks: %zu", num_chunks);


	resources.mode = AES_GCM_MODE_DECRYPT;
	result = allocate_aes_gcm_resources(aes_gcm_cfg.pci_address, num_chunks + 1, &resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to allocate AES-GCM resources: %s", doca_error_get_descr(result));
	}
	*/
        



	gettimeofday(&start, NULL);

	uint8_t *output_data = aes_gcm_decrypt(&aes_gcm_cfg, file_data, file_size, output_size, &resources);


	gettimeofday(&end, NULL);
    double time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
    //printf("Decryption time taken: %.6f seconds\n", time_taken);
    

	

	//DOCA_LOG_INFO("Decryption finished successfully");
	return output_data;

}
