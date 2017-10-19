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


/**
 * @file
 * @brief uNVMe core header file.
 */

#ifndef _UNVME_CORE_H
#define _UNVME_CORE_H

#include <sys/types.h>

#include "unvme_log.h"
#include "unvme_vfio.h"
#include "unvme_nvme.h"
#include "unvme_lock.h"
#include "unvme.h"

/// Page size
typedef char unvme_page_t[4096];

/// IO memory allocation tracking info
typedef struct _unvme_iomem {
    vfio_dma_t**            map;        ///< dynamic array of allocated memory
    int                     size;       ///< array size
    int                     count;      ///< array count
    unvme_lock_t            lock;       ///< map access lock
} unvme_iomem_t;

/// IO full descriptor
typedef struct _unvme_desc {
    void*                   buf;        ///< buffer
    u64                     slba;       ///< starting lba
    u32                     nlb;        ///< number of blocks
    u32                     qid;        ///< queue id
    u32                     opc;        ///< op code
    u32                     id;         ///< descriptor id
    void*                   sentinel;   ///< sentinel check
    struct _unvme_ioq*      ioq;        ///< IO queue context owner
    struct _unvme_desc*     prev;       ///< previous descriptor node
    struct _unvme_desc*     next;       ///< next descriptor node
    int                     error;      ///< error status
    int                     cidcount;   ///< number of pending cids
    u64                     cidmask[];  ///< cid pending bit mask
} unvme_desc_t;

/// IO queue entry
typedef struct _unvme_ioq {
    nvme_queue_t            nvmeq;      ///< NVMe associated queue
    vfio_dma_t*             sqdma;      ///< submission queue mem
    vfio_dma_t*             cqdma;      ///< completion queue mem
    vfio_dma_t*             prplist;    ///< PRP list
    u16                     cid;        ///< next cid to check and use
    int                     cidcount;   ///< number of pending cids
    int                     desccount;  ///< number of pending descriptors
    int                     masksize;   ///< bit mask size to allocate
    u64*                    cidmask;    ///< cid pending bit mask
    unvme_desc_t*           desclist;   ///< use descriptor list
    unvme_desc_t*           descfree;   ///< free descriptor list
    unvme_desc_t*           descnext;   ///< next pending descriptor to process
} unvme_ioq_t;

/// Device context
typedef struct _unvme_device {
    vfio_device_t           vfiodev;    ///< VFIO device
    nvme_device_t           nvmedev;    ///< NVMe device
    vfio_dma_t*             asqdma;     ///< admin submission queue mem
    vfio_dma_t*             acqdma;     ///< admin completion queue mem
    unvme_iomem_t           iomem;      ///< IO memory tracker
    unvme_ns_t              ns;         ///< controller namespace (id=0)
    int                     refcount;   ///< reference count
    unvme_ioq_t*            ioqs;       ///< pointer to IO queues
} unvme_device_t;

/// Session context
typedef struct _unvme_session {
    struct _unvme_session*  prev;       ///< previous session node
    struct _unvme_session*  next;       ///< next session node
    unvme_device_t*         dev;        ///< device context
    unvme_ns_t              ns;         ///< namespace
} unvme_session_t;

unvme_ns_t* unvme_do_open(int pci, int nsid, int qcount, int qsize);
int unvme_do_close(const unvme_ns_t* ns);
void* unvme_do_alloc(const unvme_ns_t* ns, u64 size);
int unvme_do_free(const unvme_ns_t* ses, void* buf);
int unvme_do_poll(unvme_desc_t* desc, int sec, u32* cqe_cs);
unvme_desc_t* unvme_rw(const unvme_ns_t* ns, int qid, int opc, void* buf, u64 slba, u32 nlb);

#endif  // _UNVME_CORE_H

