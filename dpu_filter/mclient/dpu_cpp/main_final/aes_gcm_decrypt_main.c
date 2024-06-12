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
uint8_t* aes_gcm_decrypt(struct aes_gcm_cfg *cfg, char *file_data, size_t file_size, size_t* output_size, struct aes_gcm_resources *resources, uint8_t* dst_buffer);

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


void stop_mmap(){


	doca_error_t tmp_result;
    struct program_core_objects *state = resources.state;

	if (state->buf_inv != NULL) {
		tmp_result = doca_buf_inventory_destroy(state->buf_inv);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to destroy buf inventory: %s", doca_error_get_descr(tmp_result));
		}
		state->buf_inv = NULL;
	}


	if (state->dst_mmap != NULL) {
		tmp_result = doca_mmap_stop(state->dst_mmap);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to destroy destination mmap: %s", doca_error_get_descr(tmp_result));
		}
		state->dst_mmap = NULL;
	}

	if (state->src_mmap != NULL) {
		tmp_result = doca_mmap_stop(state->src_mmap);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to destroy source mmap: %s", doca_error_get_descr(tmp_result));
		}
		state->src_mmap = NULL;
	}

	if (state->dst_mmap != NULL) {
		tmp_result = doca_mmap_destroy(state->dst_mmap);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to destroy destination mmap: %s", doca_error_get_descr(tmp_result));
		}
		state->dst_mmap = NULL;
	}

	if (state->src_mmap != NULL) {
		tmp_result = doca_mmap_destroy(state->src_mmap);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to destroy source mmap: %s", doca_error_get_descr(tmp_result));
		}
		state->src_mmap = NULL;
	}

}
uint8_t* prep_doca_buffer_dst(size_t file_size) {

	doca_error_t result;
    struct program_core_objects *state = resources.state;
	uint8_t *dst_buffer = NULL;

	size_t buffer_size = 0;
	uint64_t max_decrypt_buf_size = 0;
	size_t output_size = 0;

	buffer_size = MAX_BUFFER_SIZE;

    	

	size_t num_chunks = (file_size + buffer_size - 1) / buffer_size;
	output_size = file_size - num_chunks * aes_gcm_cfg.tag_size;

    
    result = create_core_objects(state, num_chunks+1);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create DOCA core objects: %s", doca_error_get_descr(result));
	}



	result = doca_aes_gcm_cap_task_decrypt_get_max_buf_size(doca_dev_as_devinfo(state->dev), &max_decrypt_buf_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to query AES-GCM decrypt max buf size: %s", doca_error_get_descr(result));
		return NULL;
	}

	dst_buffer = calloc(1, output_size);
	if (dst_buffer == NULL) {
		result = DOCA_ERROR_NO_MEMORY;
		DOCA_LOG_ERR("Failed to allocate memory: %s", doca_error_get_descr(result));
		return NULL;
	}


    //fast
	result = doca_mmap_set_memrange(state->dst_mmap, dst_buffer, output_size);
	//result = doca_mmap_set_memrange(state->dst_mmap, dst_buffer, max_decrypt_buf_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set mmap memory range: %s", doca_error_get_descr(result));
		return NULL;
	}


    //takes 116 msec
	result = doca_mmap_start(state->dst_mmap);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start mmap: %s", doca_error_get_descr(result));
		return NULL;
	}

	return dst_buffer;

}

uint8_t* prep_doca_buffer_src(size_t file_size, char* file_data) {
    doca_error_t result;
    struct program_core_objects *state = resources.state;

    result = doca_mmap_set_memrange(state->src_mmap, file_data, file_size);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to set mmap memory range: %s", doca_error_get_descr(result));
        return NULL;
    }

    result = doca_mmap_start(state->src_mmap);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to start mmap: %s", doca_error_get_descr(result));
        return NULL;
    }

    return (uint8_t*)file_data;  // Return the file_data cast to uint8_t*
}



uint8_t* decrypt_buffer(char* file_data, size_t file_size, size_t* output_size,  uint8_t* dst_buffer)
{
	
	
	int exit_status = EXIT_FAILURE;
	struct timeval start, end;
	doca_error_t result;
	
    size_t buffer_size = 0;


	DOCA_LOG_INFO("Start sample");


	gettimeofday(&start, NULL);

	uint8_t *output_data = aes_gcm_decrypt(&aes_gcm_cfg, file_data, file_size, output_size, &resources, dst_buffer);

	gettimeofday(&end, NULL);
    double time_taken = (end.tv_sec - start.tv_sec) * 1e6;
    time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
    //printf("Decryption time taken: %.6f seconds\n", time_taken);
    

	//DOCA_LOG_INFO("Decryption finished successfully");
	return output_data;

}
