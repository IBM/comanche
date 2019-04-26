/*
   Copyright [2017] [IBM Corporation]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/* Copyright (C) 2016, 2017 IBM Research
 *
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */
#include "self_tests.h"
#include <common/dump_utils.h>
#include <common/utils.h>
#include <rte_malloc.h>
#include <spdk/nvme.h>
#include "config.h"

static void simple_io_complete(void* arg,
                               const struct spdk_nvme_cpl* completion) {
  *((int*) arg) = 1;
}

static constexpr float CPU_FREQ = 2400.0;

void test_random_block(struct spdk_nvme_ns* ns, struct spdk_nvme_qpair* qpair,
                       void* buffer, size_t lba_max, size_t iterations,
                       bool write_flag) {
  int rc;
  int complete = 0;
  cpu_time_t start = rdtsc();
  cpu_time_t submit_cycles = 0;
  cpu_time_t completion_cycles = 0;

  PINF("test_random_block: (iter=%ld,lba_max=%ld,wf=%s)", iterations, lba_max,
       write_flag ? "true" : "false");
  for (size_t i = 0; i < iterations; i++) {
    uint64_t block = genrand64_int64() % lba_max;
  //    printf("%ld\n",block);
  retry:
    cpu_time_t submit_start = rdtsc();
    if (write_flag)
      rc = spdk_nvme_ns_cmd_write(ns, qpair, buffer,
                                  block,  // uint64_t lba,
                                  1,      // uint32_t lba_count,
                                  &simple_io_complete, &complete,
                                  0);  // uint32_t io_flags);
    else
      rc = spdk_nvme_ns_cmd_read(ns, qpair, buffer,
                                 block,  // uint64_t lba,
                                 1,      // uint32_t lba_count,
                                 &simple_io_complete, &complete,
                                 0);  // uint32_t io_flags);
    if (rc != 0) {
      PERR("nvme_ns_cmd_write failed: rc=%d", rc);
      sleep(1);
      goto retry;
    }

    submit_cycles += rdtsc() - submit_start;

    cpu_time_t compl_start = rdtsc();
    while (!complete) {
      spdk_nvme_qpair_process_completions(qpair, 0 /* unlimited completions */);
    }
    completion_cycles += rdtsc() - compl_start;
    complete = 0;
  }
  cpu_time_t end = rdtsc();

  float time_sec = (((float) (end - start)) / CPU_FREQ) / 1000000.0f;
  PINF("mean cycles/total IOP: %ld", (end - start) / iterations);
  PINF("cycles/IOP submit: %ld", submit_cycles / iterations);
  PINF("cycles/IOP completion: %ld", completion_cycles / iterations);
  PINF("time: %f seconds", time_sec);
  PINF(
      "rate: %f MB/s",
      ((float) ((CONFIG_DEVICE_BLOCK_SIZE * iterations) / 1048576)) / time_sec);

  PINF("avg. random %s range:%ld lat:%f usec", write_flag ? "write" : "read",
       lba_max, ((float) (end - start)) / (CPU_FREQ * iterations));
}

void test_sequential_block(struct spdk_nvme_ns* ns,
                           struct spdk_nvme_qpair* qpair, void* buffer,
                           size_t num_blocks, size_t lba_max, size_t iterations,
                           bool write_flag) {
  int rc;
  int complete = 0;
  cpu_time_t start = rdtsc();

  for (size_t i = 0; i < iterations; i++) {
    uint64_t block = i % lba_max;
  //    PINF("block:%ld",block);
  retry:
    if (write_flag)
      rc = spdk_nvme_ns_cmd_write(
          ns, qpair, buffer,
          block,       // uint64_t lba,
          num_blocks,  // uint32_t lba_count, i.e. write one block
          &simple_io_complete, &complete,
          0);  // uint32_t io_flags);
    else
      rc = spdk_nvme_ns_cmd_read(ns, qpair, buffer,
                                 block,       // uint64_t lba,
                                 num_blocks,  // uint32_t lba_count,
                                 &simple_io_complete, &complete,
                                 0);  // uint32_t io_flags);
    if (rc != 0) {
      PERR("nvme_ns_cmd_write failed: rc=%d", rc);
      sleep(1);
      goto retry;
    }
    while (!complete) {
      spdk_nvme_qpair_process_completions(qpair, 0 /* unlimited completions */);
    }
    complete = 0;
  }
  cpu_time_t end = rdtsc();

  float time_sec = (((float) (end - start)) / CPU_FREQ) / 1000000.0f;
  PINF("time: %f seconds", time_sec);
  PINF("rate: %f MB/s",
       ((float) ((CONFIG_DEVICE_BLOCK_SIZE * iterations * num_blocks) /
                 1048576)) /
           time_sec);

  PINF("avg. sequential %s range:%ld lat:%f usec",
       write_flag ? "write" : "read", lba_max,
       (((float) (end - start)) / CPU_FREQ) / iterations);
}

void test_rand_sequential_block(struct spdk_nvme_ns* ns,
                                struct spdk_nvme_qpair* qpair, void* buffer,
                                size_t lba_min, size_t lba_max,
                                size_t block_count, size_t iterations,
                                bool write_flag) {
  int rc;
  int complete = 0;
  cpu_time_t start = rdtsc();

  for (size_t i = 0; i < iterations; i++) {
    uint64_t block = (i % (lba_max - block_count - lba_min)) + lba_min;
  //    PINF("block:%ld",block);
  retry:
    if (write_flag)
      rc = spdk_nvme_ns_cmd_write(ns, qpair, buffer,
                                  block,        // uint64_t lba,
                                  block_count,  // uint32_t lba_count,
                                  &simple_io_complete, &complete,
                                  0);  // uint32_t io_flags);
    else
      rc = spdk_nvme_ns_cmd_read(ns, qpair, buffer,
                                 block,        // uint64_t lba,
                                 block_count,  // uint32_t lba_count,
                                 &simple_io_complete, &complete,
                                 0);  // uint32_t io_flags);
    if (rc != 0) {
      PERR("nvme_ns_cmd_write failed: rc=%d", rc);
      sleep(1);
      goto retry;
    }
    while (!complete) {
      spdk_nvme_qpair_process_completions(qpair, 0 /* unlimited completions */);
    }
    complete = 0;
  }
  cpu_time_t end = rdtsc();

  float time_sec = (((float) (end - start)) / CPU_FREQ) / 1000000.0f;
  PINF("time: %f seconds", time_sec);
  PINF(
      "rate: %f MB/s",
      ((float) ((CONFIG_DEVICE_BLOCK_SIZE * iterations) / 1048576)) / time_sec);

  PINF("avg. rand-sequential %s range:%ld-%ld lat:%f usec",
       write_flag ? "write" : "read", lba_min, lba_max,
       ((float) (end - start)) / (CPU_FREQ * iterations));
}

void test_skip_block(struct spdk_nvme_ns* ns, struct spdk_nvme_qpair* qpair,
                     void* buffer, size_t lba_max, size_t stride,
                     size_t num_strides, size_t iterations, bool write_flag) {
  int rc;
  int complete = 0;

  for (auto strides = num_strides; strides > 0; strides--) {
    cpu_time_t start = rdtsc();
    for (size_t i = 0; i < iterations; i++) {
      unsigned curr_stride = i % strides;
      uint64_t block = 1024 + (stride * curr_stride);
      //    PINF("block:%ld",block);
      assert(block < lba_max);

    retry:
      if (write_flag)
        rc = spdk_nvme_ns_cmd_write(ns, qpair, buffer,
                                    block,  // uint64_t lba,
                                    1,      // uint32_t lba_count,
                                    &simple_io_complete, &complete,
                                    0);  // uint32_t io_flags);
      else
        rc = spdk_nvme_ns_cmd_read(ns, qpair, buffer,
                                   block,  // uint64_t lba,
                                   1,      // uint32_t lba_count,
                                   &simple_io_complete, &complete,
                                   0);  // uint32_t io_flags);
      if (rc != 0) {
        PERR("nvme_ns_cmd_write failed: rc=%d", rc);
        sleep(1);
        goto retry;
      }
      while (!complete) {
        spdk_nvme_qpair_process_completions(qpair,
                                            0 /* unlimited completions */);
      }
      complete = 0;
    }
    cpu_time_t end = rdtsc();

    PINF("block skip %s stride:%ld #strides:%ld lat:%f usec",
         write_flag ? "write" : "read", stride, strides,
         ((float) (end - start)) / (CPU_FREQ * iterations));
  }
}

void test_sequential_block_mixed(struct spdk_nvme_ns* ns,
                                 struct spdk_nvme_qpair* qpair, void* buffer,
                                 void* buffer2, size_t block_count,
                                 size_t lba_max, size_t iterations) {
  int rc;
  int complete = 0;
  int complete2 = 0;
  cpu_time_t start = rdtsc();

  for (size_t i = 0; i < iterations; i++) {
    uint64_t block = (i * block_count) % lba_max;
  //    PINF("block:%ld",block);
  retry:
    rc = spdk_nvme_ns_cmd_write(
        ns, qpair, buffer,
        block,        // uint64_t lba,
        block_count,  // uint32_t lba_count, i.e. write one block
        &simple_io_complete, &complete,
        0);  // uint32_t io_flags);

    if (rc != 0) {
      PERR("nvme_ns_cmd_write failed: rc=%d", rc);
      goto retry;
    }

    rc = spdk_nvme_ns_cmd_read(ns, qpair, buffer2,
                               block + block_count,  // uint64_t lba,
                               block_count,          // uint32_t lba_count,
                               &simple_io_complete, &complete2,
                               0);  // uint32_t io_flags);
    if (rc != 0) {
      PERR("nvme_ns_cmd_write failed: rc=%d", rc);
      goto retry;
    }
    while (!complete || !complete2) {
      spdk_nvme_qpair_process_completions(qpair, 0 /* unlimited completions */);
    }
    complete = 0;
    complete2 = 0;
  }
  cpu_time_t end = rdtsc();

  float time_sec = (((float) (end - start)) / CPU_FREQ) / 1000000.0f;
  PINF("time: %f seconds", time_sec);
  PINF("rate: %f MB/s",
       ((float) ((CONFIG_DEVICE_BLOCK_SIZE * block_count * iterations) /
                 1048576)) /
           time_sec);
}

void test_metadata(struct spdk_nvme_ns* ns, struct spdk_nvme_qpair* qpair)

{
  int rc;
  int complete = 0;

  PLOG("running self_test::meta_data...");
  void* payload = rte_malloc("self-test-payload", 4096 * 2, 8);
  memset(payload, 0xFF, 4096 * 2);
  memset(payload, 0xAA, 4096);

  void* md = rte_malloc("self-test-md", 128, 64);
  memset(md, 0xCC, 128);

  PLOG("PAYLOAD:");
  hexdump(payload, 128);
  PLOG("MD:");
  hexdump(md, 128);
  PLOG("MD:");
  hexdump(&((char*) payload)[4096], 128);

  rc = spdk_nvme_ns_cmd_write_with_md(
      ns, qpair, payload, NULL,
      1,  // uint64_t lba,
      2,  // uint32_t lba_count, i.e. write one block
      &simple_io_complete, &complete,
      0,  // uint32_t io_flags);
      0, 0);
  PLOG("rc=%d", rc);

  while (!complete) spdk_nvme_qpair_process_completions(qpair, 0);
  complete = 0;

  /* clear memory */
  memset(payload, 0x0, 4096 * 2);
  memset(md, 0x0, 128);

  PLOG("read back..");

  rc = spdk_nvme_ns_cmd_read_with_md(ns, qpair, payload, NULL,
                                     1,  // uint64_t lba,
                                     1,  // uint32_t lba_count
                                     &simple_io_complete, &complete, 0, 0, 0);

  while (!complete) spdk_nvme_qpair_process_completions(qpair, 0);

  /* dump what is read back */
  PLOG("PAYLOAD:");
  hexdump(payload, 128);
  PLOG("MD:");
  hexdump(md, 128);
  PLOG("TAIL:");
  hexdump(&((char*) payload)[4096], 128);

  rte_free(md);
  rte_free(payload);
  PLOG("done.");

  asm("int3");
}
