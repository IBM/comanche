#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_ctx.h>
#include <doca_aes_gcm.h>
#include <doca_error.h>
#include <doca_log.h>

#include "common.h"
#include "aes_gcm_common.h"
#include <sys/time.h>
#include <sys/mman.h>
#include <time.h>

DOCA_LOG_REGISTER(AES_GCM_DECRYPT);

#define MAX_BUFFER_SIZE 1048576//2097152  // Define the maximum buffer size, 2MB

double get_time_diff(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_nsec - start->tv_nsec) / 1000000.0;
}


/*
 * Run aes_gcm_decrypt sample
 *
 * @cfg [in]: Configuration parameters
 * @file_data [in]: file data for the decrypt task
 * @file_size [in]: file size
 * @return: DOCA_SUCCESS on success, DOCA_ERROR otherwise.
 */
doca_error_t
aes_gcm_decrypt(struct aes_gcm_cfg *cfg, char *file_data, size_t file_size)
{
	struct aes_gcm_resources resources = {0};
	struct program_core_objects *state = NULL;
	struct doca_buf *src_doca_buf = NULL;
	struct doca_buf *dst_doca_buf = NULL;
	/* The sample will use 2 doca buffers */
	uint32_t max_bufs = 2;
	char *dst_buffer = NULL;
	uint8_t *resp_head = NULL;
	size_t data_len = 0;
	char *dump = NULL;
	FILE *out_file = NULL;
	struct doca_aes_gcm_key *key = NULL;
	doca_error_t result = DOCA_SUCCESS;
	doca_error_t tmp_result = DOCA_SUCCESS;
	uint64_t max_decrypt_buf_size = 0;
	struct timeval start_time, end_time, st_time, e_time;
    double time_spent, total_time, t_spent, t_time, time_taken = 0.0;
	size_t buffer_size = 0;
	struct timespec start, end;

    uint8_t *output_data = NULL; // Internal declaration
    size_t output_size = 0; // Manage the output size internally

   


	/*out_file = fopen(cfg->output_path, "wb");
	if (out_file == NULL) {
		DOCA_LOG_ERR("Unable to open output file: %s", cfg->output_path);
		return DOCA_ERROR_NO_MEMORY;
	}*/

    //Max size that the crypto engine supports
	buffer_size = MAX_BUFFER_SIZE;

	// Calculate total buffer size
    size_t num_chunks = (file_size + buffer_size - 1) / buffer_size;
	DOCA_LOG_INFO("Num chunks: %zu", num_chunks);


    output_size = file_size + num_chunks * cfg->tag_size;
	
       // Allocate memory for output data
    output_data = (uint8_t*) calloc(1, output_size);
    if (!output_data) {
        DOCA_LOG_ERR("Failed to allocate memory for output data");
        destroy_aes_gcm_resources(&resources);
        return DOCA_ERROR_NO_MEMORY;
    }

	//try to mlock the output
	/*if (mlock(output_data, output_size) != 0) {
    	DOCA_LOG_INFO("can't mlock");
    	goto destroy_dst_buf;
	}*/

	/* Allocate resources */
	resources.mode = AES_GCM_MODE_DECRYPT;
	result = allocate_aes_gcm_resources(cfg->pci_address, num_chunks + 1, &resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to allocate AES-GCM resources: %s", doca_error_get_descr(result));
		goto close_file;
	}

	state = resources.state;
	resources.task_started = false;

	result = doca_aes_gcm_cap_task_decrypt_get_max_buf_size(doca_dev_as_devinfo(state->dev), &max_decrypt_buf_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to query AES-GCM decrypt max buf size: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}


    	/* Create DOCA AES-GCM key */
	result = doca_aes_gcm_key_create(resources.aes_gcm, cfg->raw_key, cfg->raw_key_type, &key);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create DOCA AES-GCM key: %s", doca_error_get_descr(result));
		goto destroy_dst_buf;
	}


	/* Start AES-GCM context */
	result = doca_ctx_start(state->ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start context: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}

	clock_gettime(CLOCK_MONOTONIC, &start);


	dst_buffer = calloc(1, max_decrypt_buf_size);
	if (dst_buffer == NULL) {
		result = DOCA_ERROR_NO_MEMORY;
		DOCA_LOG_ERR("Failed to allocate memory: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}


	result = doca_mmap_set_memrange(state->dst_mmap, dst_buffer, max_decrypt_buf_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set mmap memory range: %s", doca_error_get_descr(result));
		goto free_dst_buf;
	}
	result = doca_mmap_start(state->dst_mmap);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start mmap: %s", doca_error_get_descr(result));
		goto free_dst_buf;
	}



	result = doca_mmap_set_memrange(state->src_mmap, file_data, file_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to set mmap memory range: %s", doca_error_get_descr(result));
		goto free_dst_buf;
	}

	result = doca_mmap_start(state->src_mmap);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start mmap: %s", doca_error_get_descr(result));
		goto free_dst_buf;
	}

	/* Construct DOCA buffer for each address range */
	result = doca_buf_inventory_buf_get_by_addr(state->buf_inv, state->dst_mmap, dst_buffer, max_decrypt_buf_size,
						    &dst_doca_buf);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to acquire DOCA buffer representing destination buffer: %s",
			     doca_error_get_descr(result));
		goto destroy_src_buf;
	}


    /////////////////////////////////////////////////////////////////////////////////////////


    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed_ms = get_time_diff(&start, &end);
    printf("Initialization and alloc %.6f ms\n", elapsed_ms);


	clock_gettime(CLOCK_MONOTONIC, &start);


    size_t offset = 0; //for file offset
	size_t d_offset = 0, output_offset = 0;; //for dest buffer put

    while (offset < file_size) {


        size_t current_chunk_size = (file_size - offset > buffer_size) ? buffer_size : file_size - offset;


		/* Construct DOCA buffer for each address range */
		result = doca_buf_inventory_buf_get_by_addr(state->buf_inv, state->src_mmap, file_data+offset, current_chunk_size, &src_doca_buf);
		
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Unable to acquire DOCA buffer representing source buffer: %s",
					doca_error_get_descr(result));
			goto free_dst_buf;
		}

	

        //Need to reset the length of buffer to reuse it
		doca_buf_reset_data_len(dst_doca_buf);

		/* Set data length in doca buffer */
		result = doca_buf_set_data(src_doca_buf, file_data+offset, current_chunk_size);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Unable to set DOCA buffer data: %s", doca_error_get_descr(result));
			goto destroy_dst_buf;
		}

        
    
        	// Inside the loop before calling submit_aes_gcm_encrypt_task
    	gettimeofday(&start_time, NULL);


		/* Submit AES-GCM decrypt task */
		result = submit_aes_gcm_decrypt_task(&resources, src_doca_buf, dst_doca_buf, key, (uint8_t *)cfg->iv,
							cfg->iv_length, cfg->tag_size, cfg->aad_size);
		if (result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("AES-GCM decrypt task failed: %s", doca_error_get_descr(result));
			goto destroy_key;
		}

	

		// Calculate the time taken and accumulate it
		gettimeofday(&end_time, NULL);
		time_spent = (end_time.tv_sec - start_time.tv_sec) * 1000.0;      // convert sec to ms
		time_spent += (end_time.tv_usec - start_time.tv_usec) / 1000.0;   // convert us to ms
		total_time += time_spent;

		/* Write the result to output file */
		doca_buf_get_head(dst_doca_buf, (void **)&resp_head);
		doca_buf_get_data_len(dst_doca_buf, &data_len);
		//fwrite(resp_head, sizeof(uint8_t), data_len, out_file);
		//DOCA_LOG_INFO("File was decrypted successfully and saved in: %s", cfg->output_path);

		memcpy(output_data + output_offset, resp_head, data_len);
		output_offset += data_len;

		/* Print destination buffer data */
		/*dump = hex_dump(resp_head, data_len);
		if (dump == NULL) {
			DOCA_LOG_ERR("Failed to allocate memory for printing buffer content\n");
			result = DOCA_ERROR_NO_MEMORY;
			goto destroy_key;
		}

		DOCA_LOG_INFO("AES-GCM decrypted data:\n%s", dump);
		free(dump);*/

		offset += current_chunk_size;

        //Need to refcount src buffer as we are using different buffers
		tmp_result = doca_buf_dec_refcount(src_doca_buf, NULL);
		if (tmp_result != DOCA_SUCCESS) {
			DOCA_LOG_ERR("Failed to decrease DOCA source buffer reference count: %s",
					doca_error_get_descr(tmp_result));
			DOCA_ERROR_PROPAGATE(result, tmp_result);
		}
	}

	printf("Total time spent to decrypt: %.2f ms\n", total_time);

	clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_ms = get_time_diff(&start, &end);
    printf("The loop %.6f ms\n", elapsed_ms);

    free(output_data); // Assume data is processed and no longer needed
   
    clock_gettime(CLOCK_MONOTONIC, &start);
	//free(output_data); // Assume data is processed and no longer needed

  
	tmp_result = doca_buf_dec_refcount(dst_doca_buf, NULL);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to decrease DOCA destination buffer reference count: %s",
			     doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}





	doca_task_free(doca_aes_gcm_task_decrypt_as_task(resources.decrypt_task));

    doca_ctx_stop(state->ctx); //this alone takes 323 msec


destroy_key:
	tmp_result = doca_aes_gcm_key_destroy(key);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy DOCA AES-GCM key: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
destroy_dst_buf:
	/*tmp_result = doca_buf_dec_refcount(dst_doca_buf, NULL);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to decrease DOCA destination buffer reference count: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}*/
destroy_src_buf:
	/*tmp_result = doca_buf_dec_refcount(src_doca_buf, NULL);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to decrease DOCA source buffer reference count: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}*/
free_dst_buf:
	free(dst_buffer);
destroy_resources:
	tmp_result = destroy_aes_gcm_resources(&resources);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy AES-GCM resources: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
close_file:
	//fclose(out_file);

	clock_gettime(CLOCK_MONOTONIC, &end);
	
    elapsed_ms = get_time_diff(&start, &end);
    printf("Clean and destroy: %.6f ms\n", elapsed_ms);

	return result;
}
