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



#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <common/logging.h>
#include <rte_config.h>
#include <rte_mempool.h>
#include <rte_malloc.h>

#include <spdk/nvme.h>
//#include "spdk/pci.h"

#include "nvme_init.h"

/* static void */
/* register_ns(struct probed_device * pd, */
/*             struct spdk_nvme_ctrlr *ctrlr, */
/*             struct spdk_nvme_ns *ns) */
/* { */
/* 	const struct spdk_nvme_ctrlr_data *cdata; */

/* 	/\* */
/* 	 * spdk_nvme_ctrlr is the logical abstraction in SPDK for an NVMe */
/* 	 *  controller.  During initialization, the IDENTIFY data for the */
/* 	 *  controller is read using an NVMe admin command, and that data */
/* 	 *  can be retrieved using spdk_nvme_ctrlr_get_data() to get */
/* 	 *  detailed information on the controller.  Refer to the NVMe */
/* 	 *  specification for more details on IDENTIFY for NVMe controllers. */
/* 	 *\/ */
/* 	cdata = spdk_nvme_ctrlr_get_data(ctrlr); */

/* 	if (!spdk_nvme_ns_is_active(ns)) { */
/* 		PINF("Controller %-20.20s (%-20.20s): Skipping inactive NS %u\n", */
/* 		       cdata->mn, cdata->sn, */
/* 		       spdk_nvme_ns_get_id(ns)); */
/* 		return; */
/* 	} */

/*   pd->ns = ns; */
/*   pd->ctrlr = ctrlr; */

/* 	PINF("  Namespace ID: %d size: %juGB", spdk_nvme_ns_get_id(ns), */
/* 	       spdk_nvme_ns_get_size(ns) / 1000000000); */
/* } */


/** 
 * Call during the probe process
 * 
 * @param cb_ctx 
 * @param dev Device descriptor
 * @param opts 
 * 
 * @return true to attach to this device
 */
bool probe_cb(void *cb_ctx,
              const struct spdk_nvme_transport_id *trid,
              struct spdk_nvme_ctrlr_opts *opts)
{
  struct probed_device * pd = (struct probed_device *) cb_ctx;

  char tmp[255];
  sprintf(tmp,"0000:%s", pd->device_id);

  bool result = (strcmp(tmp, trid->traddr)==0);
  if(result)
    PLOG("Using device: %s", trid->traddr);
  else
    PLOG("Ignoring device: %s", trid->traddr);

  return result;
}

/** 
 * Callback for spdk_nvme_probe() to report a device that has been
 * attached to the userspace NVMe driver.
 * 
 * @param cb_ctx 
 * @param dev Device descriptor
 * @param ctrlr 
 * @param opts 
 */
void attach_cb(void *cb_ctx,
               const struct spdk_nvme_transport_id *trid,
               struct spdk_nvme_ctrlr *ctrlr,
               const struct spdk_nvme_ctrlr_opts *opts)
{
  int num_ns;
  
  const struct spdk_nvme_ctrlr_data *cdata = spdk_nvme_ctrlr_get_data(ctrlr);
  struct probed_device * pd = (struct probed_device *) cb_ctx;
  
  /* entry = malloc(sizeof(struct ctrlr_entry)); */
  /* if (entry == NULL) { */
  /* 	perror("ctrlr_entry malloc"); */
  /* 	exit(1); */
  /* } */

  PINF("Attaching to NVMe device %s:%s:%s",
       trid->traddr,
       trid->trsvcid,
       trid->subnqn);
  
  snprintf((char*)pd->device_id, sizeof(pd->device_id), 
	   "%-20.20s (%-20.20s)", cdata->mn, cdata->sn);

  pd->ctrlr = ctrlr;

  /*
   * Each controller has one of more namespaces.  An NVMe namespace is basically
   *  equivalent to a SCSI LUN.  The controller's IDENTIFY data tells us how
   *  many namespaces exist on the controller.  For Intel(R) P3X00 controllers,
   *  it will just be one namespace.
   *
   * Note that in NVMe, namespace IDs start at 1, not 0.
   */
  num_ns = spdk_nvme_ctrlr_get_num_ns(ctrlr);
  PINF("Using controller %s with %d namespaces", pd->device_id, num_ns);
  
  /* for (nsid = 1; nsid <= num_ns; nsid++) { */
  /* 	register_ns(pd,ctrlr, spdk_nvme_ctrlr_get_ns(ctrlr, nsid)); */
  /* } */
  pd->ns = spdk_nvme_ctrlr_get_ns(ctrlr, 1); /* namespace 1 */
}

