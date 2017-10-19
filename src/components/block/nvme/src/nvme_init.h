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
