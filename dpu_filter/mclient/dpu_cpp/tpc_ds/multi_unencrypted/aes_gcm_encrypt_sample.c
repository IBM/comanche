#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include <doca_buf.h>
#include <doca_buf_inventory.h>
#include <doca_ctx.h>
#include <doca_aes_gcm.h>
#include <doca_error.h>
#include <doca_log.h>
#include <time.h>

#include "common.h"
#include "aes_gcm_common.h"

DOCA_LOG_REGISTER(AES_GCM_ENCRYPT);

#define MAX_BUFFER_SIZE 1048576 // 1MB buffer size

double get_time(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 + (end->tv_nsec - start->tv_nsec) / 1000000.0;
}

uint8_t* aes_gcm_encrypt(struct aes_gcm_cfg *cfg, char *file_data, size_t file_size, size_t* output_size, struct aes_gcm_resources *resources, uint8_t* dst_buffer) {
    struct program_core_objects *state = NULL;
    struct doca_buf *src_doca_buf = NULL;
    struct doca_buf **dst_doca_bufs = NULL;
    struct doca_aes_gcm_key *key = NULL;
    doca_error_t result = DOCA_SUCCESS;
    doca_error_t tmp_result = DOCA_SUCCESS;
    size_t buffer_size = MAX_BUFFER_SIZE - cfg->tag_size;
    size_t num_chunks = (file_size + buffer_size - 1) / buffer_size;
    size_t total_output_size = file_size + num_chunks * cfg->tag_size;
    size_t output_offset = 0;
    size_t offset = 0;
    size_t data_len = 0;
    
    *output_size = total_output_size;
    state = resources->state;
    resources->task_started = false;
    
    dst_doca_bufs = calloc(num_chunks, sizeof(struct doca_buf*));
    if (dst_doca_bufs == NULL) {
        DOCA_LOG_ERR("Failed to allocate memory for DOCA buffers");
        return NULL;
    }
    
    result = doca_aes_gcm_key_create(resources->aes_gcm, cfg->raw_key, cfg->raw_key_type, &key);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to create DOCA AES-GCM key: %s", doca_error_get_descr(result));
        free(dst_doca_bufs);
        return NULL;
    }
    
    result = doca_buf_inventory_buf_get_by_addr(state->buf_inv, state->src_mmap, file_data, file_size, &src_doca_buf);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Unable to acquire DOCA buffer representing source buffer: %s", doca_error_get_descr(result));
        free(dst_doca_bufs);
        return NULL;
    }
    
    for (uint32_t i = 0; i < num_chunks; i++) {
        size_t chunk_size = (file_size - i * buffer_size > buffer_size) ? buffer_size : file_size - i * buffer_size;
        result = doca_buf_inventory_buf_get_by_addr(state->buf_inv, state->dst_mmap, dst_buffer + i * buffer_size, chunk_size, &dst_doca_bufs[i]);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Unable to acquire DOCA buffer for destination buffer: %s", doca_error_get_descr(result));
            free(dst_doca_bufs);
            return NULL;
        }
    }
    
    
    for (uint32_t i = 0; i < num_chunks; i++) {
        size_t current_chunk_size = (file_size - offset > buffer_size) ? buffer_size : file_size - offset;
      
        result = doca_buf_set_data(src_doca_buf, file_data + offset, current_chunk_size);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("Unable to set DOCA buffer data: %s", doca_error_get_descr(result));
            goto clean;
        }
        
        result = submit_aes_gcm_encrypt_task(resources, src_doca_buf, dst_doca_bufs[i], key, (uint8_t *)cfg->iv, cfg->iv_length, cfg->tag_size, cfg->aad_size);
        if (result != DOCA_SUCCESS) {
            DOCA_LOG_ERR("AES-GCM encrypt task failed: %s", doca_error_get_descr(result));
            goto clean;
        }
        
        doca_buf_get_data_len(dst_doca_bufs[i], &data_len);
        output_offset += data_len;
        offset += current_chunk_size;
       
    }
    
    *output_size = output_offset;
    
clean:
    tmp_result = doca_buf_dec_refcount(src_doca_buf, NULL);
    if (tmp_result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to decrease DOCA source buffer reference count: %s", doca_error_get_descr(tmp_result));
        DOCA_ERROR_PROPAGATE(result, tmp_result);
    }
    
    if (dst_doca_bufs != NULL) {
        for (uint32_t i = 0; i < num_chunks; i++) {
            if (dst_doca_bufs[i] != NULL) {
                doca_buf_dec_refcount(dst_doca_bufs[i], NULL);
            }
        }
        free(dst_doca_bufs);
    }
    
    tmp_result = doca_aes_gcm_key_destroy(key);
    if (tmp_result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to destroy DOCA AES-GCM key: %s", doca_error_get_descr(tmp_result));
        DOCA_ERROR_PROPAGATE(result, tmp_result);
    }
    
    return dst_buffer;
}
