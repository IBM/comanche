/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __NVME_BLOCK_SERVICE_H__
#define __NVME_BLOCK_SERVICE_H__

#if defined(__cplusplus)
extern "C" {
#endif

struct probed_device {
  struct spdk_nvme_ctrlr *ctrlr;
  struct spdk_nvme_ns *ns;
  char device_id[1024];
};

bool probe_cb(void *cb_ctx,
              const struct spdk_nvme_transport_id *trid,
              struct spdk_nvme_ctrlr_opts *opts);

void cleanup(void);

void attach_cb(void *cb_ctx,
               const struct spdk_nvme_transport_id *trid,
               struct spdk_nvme_ctrlr *ctrlr,
               const struct spdk_nvme_ctrlr_opts *opts);


#if defined(__cplusplus)
}
#endif

#endif  // __NVME_BLOCK_SERVICE_H__
