/*
 * (C) Copyright IBM Corporation 2016. All rights reserved.
 *
 * U.S. Government Users Restricted Rights - Use, duplication or disclosure
 * restricted by GSA ADP Schedule Contract with IBM Corp.
 *
 */

/* Copyright (C) 2016, IBM Research
 *
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __DEVICE_SELF_TESTS_H__
#define __DEVICE_SELF_TESTS_H__

#include <common/cycles.h>
#include <common/rand.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "nvme_device.h"

/**
 * Random IO operations over range 0-lbamax blocks. Each operation is single
 * block count.
 *
 * @param ns Namespace pointer
 * @param qpair Queue pointer
 * @param buffer Data buffer to read/write
 * @param lba_max Max logical block address
 * @param iterations Number of iterations to perform
 * @param write_flag True for write, false for read.
 */
void test_random_block(struct spdk_nvme_ns *ns, struct spdk_nvme_qpair *qpair, void *buffer, size_t lba_max,
                       size_t iterations, bool write_flag);

/**
 * Perform sequential IO operations from block 0 to lba_max
 *
 * @param ns Namespace pointer
 * @param qpair Queue pointer
 * @param buffer Data buffer to read/write
 * @param num_blocks Number of blocks per IO operation
 * @param lba_max Max logical block address
 * @param iterations Number of iterations to perform
 * @param write_flag True for write, false for read
 */
void test_sequential_block(struct spdk_nvme_ns *ns, struct spdk_nvme_qpair *qpair, void *buffer, size_t num_blocks,
                           size_t lba_max, size_t iterations, bool write_flag);

/**
 * Perform IO operations which increment offsets (stride) N times.
 *
 * @param ns Namespace pointer
 * @param qpair Queue pointer
 * @param buffer Data buffer to read/write
 * @param lba_max Maximum logical block address of range
 * @param stride Size of stride in blocks
 * @param num_strides Number of strides for each iteration
 * @param iterations Number of iterations
 * @param write_flag True for write, false for read
 */
void test_skip_block(struct spdk_nvme_ns *ns, struct spdk_nvme_qpair *qpair, void *buffer, size_t lba_max,
                     size_t stride, size_t num_strides, size_t iterations, bool write_flag);

/**
 * Perform IO operations on randomly placed "strips" (i.e. contiguous blocks)
 *
 * @param ns Namespace pointer
 * @param qpair Queue pointer
 * @param buffer Data buffer to read/write
 * @param lba_min Minimum logical block address for range
 * @param lba_max Maximum logical block address for range
 * @param block_count Number of block in a strip
 * @param iterations Number of iterations to perform
 * @param write_flag True for write, false for read
 */
void test_rand_sequential_block(struct spdk_nvme_ns *ns, struct spdk_nvme_qpair *qpair, void *buffer, size_t lba_min,
                                size_t lba_max, size_t block_count, size_t iterations, bool write_flag);

/**
 * Mixed r/w paging style
 *
 * @param ns
 * @param qpair
 * @param buffer
 * @param buffer2
 * @param block_count
 * @param lba_max
 * @param iterations
 */
void test_sequential_block_mixed(struct spdk_nvme_ns *ns,
                                 struct spdk_nvme_qpair *qpair, void *buffer, void *buffer2,
                                 size_t block_count, size_t lba_max, size_t iterations);

/** 
 * Metadata support (extended LBA)
 * 
 * @param ns 
 * @param qpair 
 */
void test_metadata(struct spdk_nvme_ns *ns,
                   struct spdk_nvme_qpair *qpair);
#endif
