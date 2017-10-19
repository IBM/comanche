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
 * @brief UNVMe driver module.
 */

#include <sys/mman.h>
#include <string.h>
#include <signal.h>
#include <sched.h>

#include "rdtsc.h"
#include "unvme_core.h"

/// Doubly linked list add node
#define LIST_ADD(head, node)                                    \
            if ((head) != NULL) {                               \
                (node)->next = (head);                          \
                (node)->prev = (head)->prev;                    \
                (head)->prev->next = (node);                    \
                (head)->prev = (node);                          \
            } else {                                            \
                (node)->next = (node)->prev = (node);           \
                (head) = (node);                                \
            }

/// Doubly linked list remove node
#define LIST_DEL(head, node)                                    \
            if ((node->next) != (node)) {                       \
                (node)->next->prev = (node)->prev;              \
                (node)->prev->next = (node)->next;              \
                if ((head) == (node)) (head) = (node)->next;    \
            } else {                                            \
                (head) = NULL;                                  \
            }

/// IO descriptor debug print
#define PDEBUG(fmt, arg...) //fprintf(stderr, fmt "\n", ##arg)

/// Log and print an unrecoverable error message and exit
#define FATAL(fmt, arg...)  do { ERROR(fmt, ##arg); abort(); } while (0)


// Global static variables
static const char*      unvme_log = "/dev/shm/unvme.log";   ///< Log filename
static unvme_session_t* unvme_ses = NULL;                   ///< session list
static unvme_lock_t     unvme_lock = 0;                     ///< session lock
static u64              unvme_rdtsec;                   ///< rdtsc per second


/**
 * Get a descriptor entry by moving from the free to the use list.
 * @param   ioq     IO queue context
 * @return  the descriptor added to the use list.
 */
static unvme_desc_t* unvme_desc_get(unvme_ioq_t* ioq)
{
    unvme_desc_t* desc;

    if (ioq->descfree) {
        desc = ioq->descfree;
        LIST_DEL(ioq->descfree, desc);
    } else {
        desc = zalloc(sizeof(unvme_desc_t) + ioq->masksize);
        desc->ioq = ioq;
    }
    LIST_ADD(ioq->desclist, desc);

    if (desc == desc->next) {
        desc->id = 1;
        ioq->descnext = desc;
    } else {
        desc->id = desc->prev->id + 1;
    }
    ioq->desccount++;

    return desc;
}

/**
 * Put a descriptor entry back by moving it from the use to the free list.
 * @param   desc    descriptor
 */
static void unvme_desc_put(unvme_desc_t* desc)
{
    unvme_ioq_t* ioq = desc->ioq;

    if (ioq->descnext == desc) {
        if (desc != desc->next) ioq->descnext = desc->next;
        else ioq->descnext = NULL;
    }

    LIST_DEL(ioq->desclist, desc);
    memset(desc, 0, sizeof(unvme_desc_t) + ioq->masksize);
    desc->ioq = ioq;
    LIST_ADD(ioq->descfree, desc);

    ioq->desccount--;
}

/**
 * Process an I/O completion.
 * @param   ioq         io queue context
 * @param   timeout     timeout in seconds
 * @param   cqe_cs      CQE command specific DW0 returned
 * @return  0 if ok else NVMe error code (-1 means timeout).
 */
static int unvme_complete_io(unvme_ioq_t* ioq, int timeout, u32* cqe_cs)
{
    // wait for completion
    int err, cid;
    u64 endtsc = 0;
    do {
        cid = nvme_check_completion(&ioq->nvmeq, &err, cqe_cs);
        if (cid >= 0 || timeout == 0) break;
        if (endtsc == 0) endtsc = rdtsc() + timeout * unvme_rdtsec;
        //        else sched_yield();
    } while (rdtsc() < endtsc);
    if (cid < 0) return cid; // no completion
    
    // find the pending cid in the descriptor list to clear it
    unvme_desc_t* desc = ioq->descnext;
    int b = cid >> 6;
    u64 mask = (u64)1 << (cid & 63);
    while ((desc->cidmask[b] & mask) == 0) {
        desc = desc->next;
        if (desc == ioq->descnext) {
            ERROR("pending cid %d not found", cid);
            abort();
        }
    }
    if (err) desc->error = err;

    desc->cidmask[b] &= ~mask;
    desc->cidcount--;
    ioq->cidmask[b] &= ~mask;
    ioq->cidcount--;
    ioq->cid = cid;

    // check to advance next pending descriptor
    if (ioq->cidcount) {
        while (ioq->descnext->cidcount == 0) ioq->descnext = ioq->descnext->next;
    }
    PDEBUG("# c q%d={%d %d %#lx} d={%d %d %#lx} @%d",
           ioq->nvmeq.id, cid, ioq->cidcount, *ioq->cidmask,
           desc->id, desc->cidcount, *desc->cidmask, ioq->descnext->id);
    return err;
}

/**
 * Submit a single read/write command within the device limit.
 * @param   ns          namespace handle
 * @param   desc        descriptor
 * @param   buf         data buffer
 * @param   slba        starting lba
 * @param   nlb         number of logical blocks
 * @return  cid if ok else -1.
 */
static int unvme_submit_io(const unvme_ns_t* ns, unvme_desc_t* desc,
                           void* buf, u64 slba, u32 nlb)
{
    unvme_device_t* dev = ((unvme_session_t*)ns->ses)->dev;
    unvme_ioq_t* ioq = desc->ioq;

    if (nlb > ns->maxbpio) {
        ERROR("block count %d exceeds limit %d", nlb, ns->maxbpio);
        return -1;
    }

    // find DMA buffer address
    vfio_dma_t* dma = NULL;
    unvme_lockr(&dev->iomem.lock);
    int i;
    for (i = 0; i < dev->iomem.count; i++) {
        dma = dev->iomem.map[i];
        if (dma->buf <= buf && buf < (dma->buf + dma->size)) break;
    }
    unvme_unlockr(&dev->iomem.lock);
    if (i == dev->iomem.count) {
        ERROR("invalid I/O buffer address");
        return -1;
    }

    u64 addr = dma->addr + (u64)(buf - dma->buf);
    if ((addr & (ns->blocksize - 1)) != 0) {
        ERROR("unaligned buffer address");
        return -1;
    }
    if ((addr + nlb * ns->blocksize) > (dma->addr + dma->size)) {
        ERROR("buffer overrun");
        return -1;
    }

    // find a free cid
    // if submission queue is full then process a pending entry first
    u16 cid;
    if ((ioq->cidcount + 1) < ns->qsize) {
        cid = ioq->cid;
        while (ioq->cidmask[cid >> 6] & ((u64)1 << (cid & 63))) {
            if (++cid >= ns->qsize) cid = 0;
        }
        ioq->cid = cid;
    } else {
        // if process completion error, clear the current pending descriptor
        unvme_desc_t* desc = ioq->descnext;
        int err = unvme_complete_io(ioq, UNVME_TIMEOUT, NULL);
        if (err != 0) {
            if (err == -1) {
                ERROR("q%d timeout", ioq->nvmeq.id);
                abort();
            }
            while (desc->cidcount) {
                if (unvme_complete_io(ioq, UNVME_TIMEOUT, NULL) == -1) {
                    ERROR("q%d timeout", ioq->nvmeq.id);
                    abort();
                }
            }
        }
        cid = ioq->cid;
    }

    // compose PRPs based on cid
    int numpages = (nlb + ns->nbpp - 1) >> ns->bpshift;
    u64 prp1 = addr;
    u64 prp2 = 0;
    if (numpages == 2) {
        prp2 = addr + ns->pagesize;
    } else if (numpages > 2) {
        int prpoff = cid << ns->pageshift;
        u64* prplist = ioq->prplist->buf + prpoff;
        prp2 = ioq->prplist->addr + prpoff;
        for (i = 1; i < numpages; i++) {
            addr += ns->pagesize;
            *prplist++ = addr;
        }
    }

    if (nvme_cmd_rw(&ioq->nvmeq, desc->opc, cid,
                    ns->id, slba, nlb, prp1, prp2) == 0) {
        int b = cid >> 6;
        u64 mask = (u64)1 << (cid & 63);
        ioq->cidmask[b] |= mask;
        ioq->cidcount++;
        desc->cidmask[b] |= mask;
        desc->cidcount++;
        PDEBUG("# %c %#lx %#x q%d={%d %d %#lx} d={%d %d %#lx}",
               desc->opc == NVME_CMD_READ ? 'r' : 'w', slba, nlb,
               ioq->nvmeq.id, cid, ioq->cidcount, *ioq->cidmask,
               desc->id, desc->cidcount, *desc->cidmask);
        return cid;
    }
    return -1;
}

/**
 * Setup admin queue.
 * @param   dev         device context
 * @param   qsize       admin queue depth
 */
static void unvme_adminq_create(unvme_device_t* dev, int qsize)
{
    DEBUG_FN("%x qd=%d", dev->vfiodev.pci, qsize);
    dev->asqdma = vfio_dma_alloc(&dev->vfiodev, qsize * sizeof(nvme_sq_entry_t));
    dev->acqdma = vfio_dma_alloc(&dev->vfiodev, qsize * sizeof(nvme_cq_entry_t));
    if (!dev->asqdma || !dev->acqdma)
        FATAL("vfio_dma_alloc");
    if (!nvme_setup_adminq(&dev->nvmedev, qsize,
                           dev->asqdma->buf, dev->asqdma->addr,
                           dev->acqdma->buf, dev->acqdma->addr))
        FATAL("nvme_setup_adminq failed");
}

/**
 * Delete admin queue.
 * @param   dev         device context
 */
static void unvme_adminq_delete(unvme_device_t* dev)
{
    DEBUG_FN("%x", dev->vfiodev.pci);
    if (dev->asqdma) vfio_dma_free(dev->asqdma);
    if (dev->acqdma) vfio_dma_free(dev->acqdma);
}

/**
 * Create an I/O queue.
 * @param   dev         device context
 * @param   q           queue index starting at 0
 */
static void unvme_ioq_create(unvme_device_t* dev, int q)
{
    unvme_ioq_t* ioq = dev->ioqs + q;
    int qsize = dev->ns.qsize;
    ioq->sqdma = vfio_dma_alloc(&dev->vfiodev, qsize * sizeof(nvme_sq_entry_t));
    ioq->cqdma = vfio_dma_alloc(&dev->vfiodev, qsize * sizeof(nvme_sq_entry_t));
    if (!ioq->sqdma || !ioq->cqdma)
        FATAL("vfio_dma_alloc");
    if (!nvme_create_ioq(&dev->nvmedev, &ioq->nvmeq, q + 1, qsize,
                         ioq->sqdma->buf, ioq->sqdma->addr,
                         ioq->cqdma->buf, ioq->cqdma->addr))
        FATAL("nvme_create_ioq %d failed", q + 1);

    // setup descriptors and pending masks
    ioq->masksize = ((qsize + 63) >> 6) << 3; // ((qsize + 63) / 64) * sizeof(u64);
    ioq->cidmask = zalloc(ioq->masksize);
    int i;
    for (i = 0; i < 16; i++) unvme_desc_get(ioq);
    ioq->descfree = ioq->desclist;
    ioq->desclist = NULL;
    ioq->desccount = 0;

    // allocate PRP list pages (assume maxppio fits in 1 PRP list page)
    ioq->prplist = vfio_dma_alloc(&dev->vfiodev, qsize << dev->ns.pageshift);
    if (!ioq->prplist)
        FATAL("vfio_dma_alloc");

    DEBUG_FN("%x q=%d qd=%d db=%#04lx", dev->vfiodev.pci, ioq->nvmeq.id, qsize,
             (u64)ioq->nvmeq.sq_doorbell - (u64)dev->nvmedev.reg);
}

/**
 * Delete an I/O queue.
 * @param   dev         device context
 * @param   q           queue index starting at 0
 */
static void unvme_ioq_delete(unvme_device_t* dev, int q)
{
    DEBUG_FN("%x %d", dev->vfiodev.pci, q + 1);
    unvme_ioq_t* ioq = &dev->ioqs[q];

    // free all descriptors
    unvme_desc_t* desc;
    while ((desc = ioq->desclist) != NULL) {
        LIST_DEL(ioq->desclist, desc);
        free(desc);
    }
    while ((desc = ioq->descfree) != NULL) {
        LIST_DEL(ioq->descfree, desc);
        free(desc);
    }

    if (ioq->cidmask) free(ioq->cidmask);
    if (ioq->prplist) vfio_dma_free(ioq->prplist);
    if (ioq->cqdma) vfio_dma_free(ioq->cqdma);
    if (ioq->sqdma) vfio_dma_free(ioq->sqdma);
    memset(ioq, 0, sizeof(*ioq));
}

/**
 * Initialize a namespace instance.
 * @param   ns          namespace context
 * @param   nsid        namespace id
 */
static void unvme_ns_init(unvme_ns_t* ns, int nsid)
{
    unvme_device_t* dev = ((unvme_session_t*)ns->ses)->dev;
    ns->id = nsid;
    ns->maxiopq = ns->qsize - 1;

    vfio_dma_t* dma = vfio_dma_alloc(&dev->vfiodev, ns->pagesize);
    if (nvme_acmd_identify(&dev->nvmedev, nsid, dma->addr, 0))
        FATAL("nvme_acmd_identify %d failed", nsid);
    nvme_identify_ns_t* idns = (nvme_identify_ns_t*)dma->buf;
    ns->blockcount = idns->ncap;
    ns->blockshift = idns->lbaf[idns->flbas & 0xF].lbads;
    ns->blocksize = 1 << ns->blockshift;
    ns->bpshift = ns->pageshift - ns->blockshift;
    ns->nbpp = 1 << ns->bpshift;
    ns->pagecount = ns->blockcount >> ns->bpshift;
    ns->maxbpio = ns->maxppio << ns->bpshift;
    vfio_dma_free(dma);

    sprintf(ns->device + strlen(ns->device), "/%d", nsid);
    DEBUG_FN("%s qc=%d qd=%d bs=%d bc=%#lx mbio=%d", ns->device, ns->qcount,
             ns->qsize, ns->blocksize, ns->blockcount, ns->maxbpio);
}

/**
 * Clean up.
 */
static void unvme_cleanup(unvme_session_t* ses)
{
    unvme_device_t* dev = ses->dev;
    if (--dev->refcount == 0) {
        DEBUG_FN("%s", ses->ns.device);
        int q;
        for (q = 0; q < dev->ns.qcount; q++) unvme_ioq_delete(dev, q);
        unvme_adminq_delete(dev);
        nvme_delete(&dev->nvmedev);
        vfio_delete(&dev->vfiodev);
        free(dev->ioqs);
        free(dev);
    }
    LIST_DEL(unvme_ses, ses);
    free(ses);
    if (!unvme_ses) log_close();
}


/**
 * Open and attach to a UNVMe driver.
 * @param   pci         PCI device id
 * @param   nsid        namespace id
 * @param   qcount      number of queues (0 for max number of queues support)
 * @param   qsize       size of each queue (0 default to 65)
 * @return  namespace pointer or NULL if error.
 */
unvme_ns_t* unvme_do_open(int pci, int nsid, int qcount, int qsize)
{
    unvme_lockw(&unvme_lock);
    if (!unvme_ses) {
        if (log_open(unvme_log, "w")) {
            unvme_unlockw(&unvme_lock);
            exit(1);
        }
        unvme_rdtsec = rdtsc_second();
    }

    // check for existing opened device
    unvme_session_t* xses = unvme_ses;
    while (xses) {
        if (xses->ns.pci == pci) {
            if (nsid > xses->ns.nscount) {
                ERROR("invalid %06x nsid %d (max %d)", pci, nsid, xses->ns.nscount);
                return NULL;
            }
            if (xses->ns.id == nsid) {
                ERROR("%06x nsid %d is in use", pci);
                return NULL;
            }
            break;
        }
        xses = xses->next;
        if (xses == unvme_ses) xses = NULL;
    }

    unvme_device_t* dev;
    if (xses) {
        dev = xses->dev;
    } else {
        // setup controller namespace
        dev = zalloc(sizeof(unvme_device_t));
        vfio_create(&dev->vfiodev, pci);
        nvme_create(&dev->nvmedev, dev->vfiodev.fd);
        unvme_adminq_create(dev, 8);

        // get controller info
        vfio_dma_t* dma = vfio_dma_alloc(&dev->vfiodev, 4096);
        if (nvme_acmd_identify(&dev->nvmedev, 0, dma->addr, 0))
            FATAL("nvme_acmd_identify controller failed");
        nvme_identify_ctlr_t* idc = (nvme_identify_ctlr_t*)dma->buf;
        if (nsid > idc->nn) {
            ERROR("invalid %06x nsid %d (max %d)", pci, nsid, idc->nn);
            return NULL;
        }

        unvme_ns_t* ns = &dev->ns;
        ns->pci = pci;
        ns->id = 0;
        ns->nscount = idc->nn;
        sprintf(ns->device, "%02x:%02x.%x", pci >> 16, (pci >> 8) & 0xff, pci & 0xff);
        ns->maxqsize = dev->nvmedev.maxqsize;
        ns->pageshift = dev->nvmedev.pageshift;
        ns->pagesize = 1 << ns->pageshift;
        int i;
        ns->vid = idc->vid;
        memcpy(ns->mn, idc->mn, sizeof (ns->mn));
        for (i = sizeof (ns->mn) - 1; i > 0 && ns->mn[i] == ' '; i--) ns->mn[i] = 0;
        memcpy(ns->sn, idc->sn, sizeof (ns->sn));
        for (i = sizeof (ns->sn) - 1; i > 0 && ns->sn[i] == ' '; i--) ns->sn[i] = 0;
        memcpy(ns->fr, idc->fr, sizeof (ns->fr));
        for (i = sizeof (ns->fr) - 1; i > 0 && ns->fr[i] == ' '; i--) ns->fr[i] = 0;

        // limit to 1 PRP list page (pagesize / sizeof u64)
        ns->maxppio = ns->pagesize >> 3;
        if (idc->mdts) {
            int mp = 2 << (idc->mdts - 1);
            if (ns->maxppio > mp) ns->maxppio = mp;
        }
        vfio_dma_free(dma);

        // get max number of queues supported
        nvme_feature_num_queues_t nq;
        if (nvme_acmd_get_features(&dev->nvmedev, 0,
                                   NVME_FEATURE_NUM_QUEUES, 0, 0, (u32*)&nq))
            FATAL("nvme_acmd_get_features number of queues failed");
        int maxqcount = (nq.nsq < nq.ncq ? nq.nsq : nq.ncq) + 1;
        if (qcount <= 0) qcount = maxqcount;
        if (qsize <= 1) qsize = UNVME_QSIZE;
        ns->maxqcount = maxqcount;
        ns->qcount = qcount;
        ns->qsize = qsize;

        // setup IO queues
        dev->ioqs = zalloc(qcount * sizeof(unvme_ioq_t));
        for (i = 0; i < qcount; i++) unvme_ioq_create(dev, i);
    }

    // allocate new session
    unvme_session_t* ses = zalloc(sizeof(unvme_session_t));
    ses->dev = dev;
    dev->refcount++;
    memcpy(&ses->ns, &ses->dev->ns, sizeof(unvme_ns_t));
    ses->ns.ses = ses;
    unvme_ns_init(&ses->ns, nsid);
    LIST_ADD(unvme_ses, ses);

    INFO_FN("%s (%.40s) is ready", ses->ns.device, ses->ns.mn);
    unvme_unlockw(&unvme_lock);
    return &ses->ns;
}

/**
 * Close and detach from a UNVMe driver.
 * @param   ns          namespace handle
 * @return  0 if ok else -1.
 */
int unvme_do_close(const unvme_ns_t* ns)
{
    DEBUG_FN("%s", ns->device);
    unvme_session_t* ses = ns->ses;
    if (ns->pci != ses->dev->vfiodev.pci) return -1;
    unvme_lockw(&unvme_lock);
    unvme_cleanup(ses);
    unvme_unlockw(&unvme_lock);
    return 0;
}

/**
 * Allocate an I/O buffer.
 * @param   ns          namespace handle
 * @param   size        buffer size
 * @return  the allocated buffer or NULL if failure.
 */
void* unvme_do_alloc(const unvme_ns_t* ns, u64 size)
{
    DEBUG_FN("%s %#lx", ns->device, size);
    unvme_device_t* dev = ((unvme_session_t*)ns->ses)->dev;
    unvme_iomem_t* iomem = &dev->iomem;
    void* buf = NULL;

    unvme_lockw(&iomem->lock);
    vfio_dma_t* dma = vfio_dma_alloc(&dev->vfiodev, size);
    if (dma) {
        if (iomem->count == iomem->size) {
            iomem->size += 256;
            iomem->map = realloc(iomem->map, iomem->size * sizeof(void*));
        }
        iomem->map[iomem->count++] = dma;
        buf = dma->buf;
    }
    unvme_unlockw(&iomem->lock);
    return buf;
}

/**
 * Free an I/O buffer.
 * @param   ns          namespace handle
 * @param   buf         buffer pointer
 * @return  0 if ok else -1.
 */
int unvme_do_free(const unvme_ns_t* ns, void* buf)
{
    DEBUG_FN("%s %p", ns->device, buf);
    unvme_device_t* dev = ((unvme_session_t*)ns->ses)->dev;
    unvme_iomem_t* iomem = &dev->iomem;

    unvme_lockw(&iomem->lock);
    int i;
    for (i = 0; i < iomem->count; i++) {
        if (buf == iomem->map[i]->buf) {
            vfio_dma_free(iomem->map[i]);
            iomem->count--;
            if (i != iomem->count)
                iomem->map[i] = iomem->map[iomem->count];
            unvme_unlockw(&iomem->lock);
            return 0;
        }
    }
    unvme_unlockw(&iomem->lock);
    return -1;
}

/**
 * Poll for completion status of a previous IO submission.
 * If there's no error, the descriptor will be released.
 * @param   desc        IO descriptor
 * @param   timeout     in seconds
 * @param   cqe_cs      CQE command specific DW0 returned
 * @return  0 if ok else error status.
 */
int unvme_do_poll(unvme_desc_t* desc, int timeout, u32* cqe_cs)
{
    if (desc->sentinel != desc->buf)
        FATAL("bad IO descriptor");
    PDEBUG("# POLL d={%d %d %#lx}", desc->id, desc->cidcount, *desc->cidmask);
    int err = 0;
    while (desc->cidcount) {
        if ((err = unvme_complete_io(desc->ioq, timeout, cqe_cs)) != 0) break;
    }
    if (desc->id != 0 && desc->cidcount == 0) unvme_desc_put(desc);
    PDEBUG("# q%d +%d", desc->ioq->nvmeq.id, desc->ioq->desccount);
    return err;
}

/**
 * Submit a read/write command that may require multiple I/O submissions
 * and processing some completions.
 * @param   ns          namespace handle
 * @param   qid         queue id
 * @param   opc         op code
 * @param   buf         data buffer
 * @param   slba        starting lba
 * @param   nlb         number of logical blocks
 * @return  I/O descriptor or NULL if error.
 */
unvme_desc_t* unvme_rw(const unvme_ns_t* ns, int qid, int opc,
                       void* buf, u64 slba, u32 nlb)
{
    unvme_ioq_t* ioq = ((unvme_session_t*)ns->ses)->dev->ioqs + qid;
    unvme_desc_t* desc = unvme_desc_get(ioq);
    desc->opc = opc;
    desc->buf = buf;
    desc->qid = qid;
    desc->slba = slba;
    desc->nlb = nlb;
    desc->sentinel = buf;

    PDEBUG("# %s %#lx %#x @%d +%d", opc == NVME_CMD_READ ? "READ" : "WRITE",
           slba, nlb, desc->id, ioq->desccount);
    while (nlb) {
        int n = ns->maxbpio;
        if (n > nlb) n = nlb;
        int cid = unvme_submit_io(ns, desc, buf, slba, n);
        if (cid < 0) {
            if (unvme_do_poll(desc, UNVME_TIMEOUT, NULL) != 0) {
                ERROR("q%d timeout", ioq->nvmeq.id);
                abort();
            }
            unvme_desc_put(desc);
            return NULL;
        }

        buf += n * ns->blocksize;
        slba += n;
        nlb -= n;
    }

    return desc;
}

