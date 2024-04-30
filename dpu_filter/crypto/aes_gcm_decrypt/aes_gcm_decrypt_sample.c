/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

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

DOCA_LOG_REGISTER(AES_GCM_DECRYPT);

/*
 * Run aes_gcm_decrypt sample
 *
 * @cfg [in]: Configuration parameters
 * @file_data [in]: file data for the decrypt task
 * @file_size [in]: file size
 * @return: DOCA_SUCCESS on success, DOCA_ERROR otherwise.
 */
doca_error_t
aes_gcm_decrypt(struct aes_gcm_cfg *cfg, char *file_data, size_t file_size, FILE *out_file)
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
	
	struct doca_aes_gcm_key *key = NULL;
	doca_error_t result = DOCA_SUCCESS;
	doca_error_t tmp_result = DOCA_SUCCESS;
	uint64_t max_decrypt_buf_size = 0;


	/* Allocate resources */
	resources.mode = AES_GCM_MODE_DECRYPT;
	result = allocate_aes_gcm_resources(cfg->pci_address, max_bufs, &resources);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to allocate AES-GCM resources: %s", doca_error_get_descr(result));

	}

	state = resources.state;

	result = doca_aes_gcm_cap_task_decrypt_get_max_buf_size(doca_dev_as_devinfo(state->dev), &max_decrypt_buf_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to query AES-GCM decrypt max buf size: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}

	if (file_size > max_decrypt_buf_size) {
		DOCA_LOG_ERR("File size %zu > max buffer size %zu", file_size, max_decrypt_buf_size);
		goto destroy_resources;
	}

	/* Start AES-GCM context */
	result = doca_ctx_start(state->ctx);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to start context: %s", doca_error_get_descr(result));
		goto destroy_resources;
	}

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
	result = doca_buf_inventory_buf_get_by_addr(state->buf_inv, state->src_mmap, file_data, file_size,
						    &src_doca_buf);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to acquire DOCA buffer representing source buffer: %s",
			     doca_error_get_descr(result));
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

	/* Set data length in doca buffer */
	result = doca_buf_set_data(src_doca_buf, file_data, file_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to set DOCA buffer data: %s", doca_error_get_descr(result));
		goto destroy_dst_buf;
	}

	/* Create DOCA AES-GCM key */
	result = doca_aes_gcm_key_create(resources.aes_gcm, cfg->raw_key, cfg->raw_key_type, &key);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Unable to create DOCA AES-GCM key: %s", doca_error_get_descr(result));
		goto destroy_dst_buf;
	}

	/* Submit AES-GCM decrypt task */
	result = submit_aes_gcm_decrypt_task(&resources, src_doca_buf, dst_doca_buf, key, (uint8_t *)cfg->iv,
					     cfg->iv_length, cfg->tag_size, cfg->aad_size);
	if (result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("AES-GCM decrypt task failed: %s", doca_error_get_descr(result));
		goto destroy_key;
	}

	/* Write the result to output file */
	doca_buf_get_head(dst_doca_buf, (void **)&resp_head);
	doca_buf_get_data_len(dst_doca_buf, &data_len);
	fwrite(resp_head, sizeof(uint8_t), data_len, out_file);
	DOCA_LOG_INFO("File was decrypted successfully and saved in: %s", cfg->output_path);

	/* Print destination buffer data */
	/*dump = hex_dump(resp_head, data_len);
	if (dump == NULL) {
		DOCA_LOG_ERR("Failed to allocate memory for printing buffer content\n");
		result = DOCA_ERROR_NO_MEMORY;
		goto destroy_key;
	}

	DOCA_LOG_INFO("AES-GCM decrypted data:\n%s", dump);
	free(dump);*/

destroy_key:
	tmp_result = doca_aes_gcm_key_destroy(key);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy DOCA AES-GCM key: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
destroy_dst_buf:
	tmp_result = doca_buf_dec_refcount(dst_doca_buf, NULL);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to decrease DOCA destination buffer reference count: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
destroy_src_buf:
	tmp_result = doca_buf_dec_refcount(src_doca_buf, NULL);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to decrease DOCA source buffer reference count: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}
free_dst_buf:
	free(dst_buffer);
destroy_resources:
	tmp_result = destroy_aes_gcm_resources(&resources);
	if (tmp_result != DOCA_SUCCESS) {
		DOCA_LOG_ERR("Failed to destroy AES-GCM resources: %s", doca_error_get_descr(tmp_result));
		DOCA_ERROR_PROPAGATE(result, tmp_result);
	}


	return result;
}
