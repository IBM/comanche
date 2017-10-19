/*-
 *   BSD LICENSE
 *
 *   Copyright (c) Intel Corporation.
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

  PLOG("Checking against device: %s", trid->traddr);
  char tmp[255];
  sprintf(tmp,"0000:%s", pd->device_id);

  return (strcmp(tmp, trid->traddr)==0);
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

