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
 * @brief NVMe implementation.
 */

#include <sys/mman.h>
#include <string.h>
#include <errno.h>

#include "rdtsc.h"
#include "unvme_log.h"
#include "unvme_nvme.h"


/// @cond

static inline u32 r32(nvme_device_t* dev, u32* addr)
{
    u32 val = *addr;
    DEBUG("r32 %#lx %#x", (u64) addr - (u64) dev->reg, val);
    return val;
}

static inline void w32(nvme_device_t* dev, u32* addr, u32 val)
{
    DEBUG("w32 %#lx %#x", (u64) addr - (u64) dev->reg, val);
    *addr = val;
}

static inline u64 r64(nvme_device_t* dev, u64* addr)
{
    u64 val = *addr;
    DEBUG("r64 %#lx %#lx", (u64) addr - (u64) dev->reg, val);
    return val;
}

static inline void w64(nvme_device_t* dev, u64* addr, u64 val)
{
    DEBUG("w64 %#lx %#lx", (u64) addr - (u64) dev->reg, val);
    *addr = val;
}

/// @endcond

static u64 nvme_rdtsec;                 ///< rdtsc per second


/**
 * Wait for controller enabled/disabled state.
 * @param   dev         device context
 * @param   ready       ready state (enabled/disabled)
 * @return  0 if ok else -1.
 */
static int nvme_ctlr_wait_ready(nvme_device_t* dev, int ready)
{
    int i;
    for (i = 0; i < dev->timeout; i++) {
        usleep(500000);
        nvme_controller_status_t csts;
        csts.val = r32(dev, &dev->reg->csts.val);
        if (csts.rdy == ready) return 0;
    }

    ERROR("timeout waiting for ready %d", ready);
    return -1;
}

/**
 * Disable controller.
 * @param   dev         device context
 * @return  0 if ok else -1.
 */
static int nvme_ctlr_disable(nvme_device_t* dev)
{
    DEBUG_FN();
    nvme_controller_config_t cc;
    cc.val = r32(dev, &dev->reg->cc.val);
    cc.en = 0;
    w32(dev, &dev->reg->cc.val, cc.val);
    return nvme_ctlr_wait_ready(dev, 0);
}

/**
 * Enable controller.
 * @param   dev         device context
 * @param   cc          controller configuration settings
 * @return  0 if ok else -1.
 */
static int nvme_ctlr_enable(nvme_device_t* dev, nvme_controller_config_t cc)
{
    DEBUG_FN();
    cc.en = 1;
    w32(dev, &dev->reg->cc.val, cc.val);
    return nvme_ctlr_wait_ready(dev, 1);
}

/**
 * Submit an entry at submission queue tail.
 * @param   q           queue
 * @return  0 if ok else -1.
 */
static int nvme_submit_cmd(nvme_queue_t* q)
{
    int tail = q->sq_tail;
    HEX_DUMP(&q->sq[tail], sizeof(nvme_sq_entry_t));
    if (++tail == q->size) tail = 0;
#if 0
    // Some SSD does not advance sq_head properly (e.g. Intel DC D3600)
    // so let the upper layer detect queue full error condition
    if (tail == q->sq_head) {
        ERROR("sq full at %d", tail);
        return -1;
    }
#endif
    q->sq_tail = tail;
    w32(q->dev, q->sq_doorbell, tail);
    return 0;
}

/**
 * Check a completion queue and return the completed command id and status.
 * @param   q           queue
 * @param   stat        completion status returned
 * @param   cqe_cs      CQE command specific DW0 returned
 * @return  the completed command id or -1 if there's no completion.
 */
int nvme_check_completion(nvme_queue_t* q, int* stat, u32* cqe_cs)
{
    *stat = 0;
    nvme_cq_entry_t* cqe = &q->cq[q->cq_head];
    if (cqe->p == q->cq_phase) return -1;

    *stat = cqe->psf & 0xfe;
    if (++q->cq_head == q->size) {
        q->cq_head = 0;
        q->cq_phase = !q->cq_phase;
    }
    if (cqe_cs) *cqe_cs = cqe->cs;
    w32(q->dev, q->cq_doorbell, q->cq_head);

#if 0
    // Some SSD does not advance sq_head properly (e.g. Intel DC D3600)
    // so let the upper layer detect queue full error condition
    q->sq_head = cqe->sqhd;
#endif

    if (*stat == 0) {
        DEBUG_FN("q=%d h=%d cid=%#x (C)", q->id, q->sq_head, cqe->cid);
    } else {
        ERROR("q=%d cid=%#x stat=%#x (dnr=%d m=%d sct=%d sc=%#x) (C)",
              q->id, cqe->cid, *stat, cqe->dnr, cqe->m, cqe->sct, cqe->sc);
    }

    return cqe->cid;
}

/**
 * Wait for a given command completion until timeout.
 * @param   q           queue
 * @param   cid         cid
 * @param   timeout     timeout in seconds
 * @return  completion status (0 if ok).
 */
int nvme_wait_completion(nvme_queue_t* q, int cid, int timeout)
{
    u64 endtsc = 0;

    do {
        int stat;
        int ret = nvme_check_completion(q, &stat, NULL);
        if (ret >= 0) {
            if (ret == cid && stat == 0) return 0;
            if (ret != cid) {
                ERROR("cid wait=%#x recv=%#x", cid, ret);
                stat = -1;
            }
            return stat;
        } else if (endtsc == 0) {
            endtsc = rdtsc() + timeout * nvme_rdtsec;
        }
    } while (rdtsc() < endtsc);

    ERROR("timeout");
    return -1;
}

/**
 * NVMe identify command.
 * Submit the command and wait for completion.
 * @param   dev         device context
 * @param   nsid        namespace id (< 0 implies cns)
 * @param   prp1        PRP1 address
 * @param   prp2        PRP2 address
 * @return  completion status (0 if ok).
 */
int nvme_acmd_identify(nvme_device_t* dev, int nsid, u64 prp1, u64 prp2)
{
    nvme_queue_t* adminq = &dev->adminq;
    int cid = adminq->sq_tail;
    nvme_acmd_identify_t* cmd = &adminq->sq[cid].identify;

    memset(cmd, 0, sizeof (*cmd));
    cmd->common.opc = NVME_ACMD_IDENTIFY;
    cmd->common.cid = cid;
    cmd->common.nsid = nsid;
    cmd->common.prp1 = prp1;
    cmd->common.prp2 = prp2;
    cmd->cns = nsid == 0 ? 1 : 0;

    DEBUG_FN("cid=%#x nsid=%d", cid, nsid);
    int err = nvme_submit_cmd(adminq);
    if (!err) err = nvme_wait_completion(adminq, cid, 10);
    return err;
}

/**
 * NVMe get log page command.
 * Submit the command and wait for completion.
 * @param   dev         device context
 * @param   nsid        namespace id
 * @param   lid         log page id
 * @param   numd        number of dwords
 * @param   prp1        PRP1 address
 * @param   prp2        PRP2 address
 * @return  completion status (0 if ok).
 */
int nvme_acmd_get_log_page(nvme_device_t* dev, int nsid,
                           int lid, int numd, u64 prp1, u64 prp2)
{
    nvme_queue_t* adminq = &dev->adminq;
    int cid = adminq->sq_tail;
    nvme_acmd_get_log_page_t* cmd = &adminq->sq[cid].get_log_page;

    memset(cmd, 0, sizeof (*cmd));
    cmd->common.opc = NVME_ACMD_GET_LOG_PAGE;
    cmd->common.cid = cid;
    cmd->common.nsid = nsid;
    cmd->common.prp1 = prp1;
    cmd->common.prp2 = prp2;
    cmd->lid = lid;
    cmd->numd = numd;

    DEBUG_FN("cid=%#x lid=%d", cid, lid);
    int err = nvme_submit_cmd(adminq);
    if (!err) err = nvme_wait_completion(adminq, cid, 10);
    if (err) ERROR();
    return err;
}

/**
 * NVMe get features command.
 * Submit the command and wait for completion.
 * @param   dev         device context
 * @param   nsid        namespace id
 * @param   fid         feature id
 * @param   prp1        PRP1 address
 * @param   prp2        PRP2 address
 * @param   res         dword 0 value returned
 * @return  completion status (0 if ok).
 */
int nvme_acmd_get_features(nvme_device_t* dev, int nsid,
                           int fid, u64 prp1, u64 prp2, u32* res)
{
    nvme_queue_t* adminq = &dev->adminq;
    int cid = adminq->sq_tail;
    nvme_acmd_get_features_t* cmd = &adminq->sq[cid].get_features;

    memset(cmd, 0, sizeof (*cmd));
    cmd->common.opc = NVME_ACMD_GET_FEATURES;
    cmd->common.cid = cid;
    cmd->common.nsid = nsid;
    cmd->common.prp1 = prp1;
    cmd->common.prp2 = prp2;
    cmd->fid = fid;
    *res = -1;

    DEBUG_FN("cid=%#x fid=%d", cid, fid);
    int err = nvme_submit_cmd(adminq);
    if (!err) err = nvme_wait_completion(adminq, cid, 10);
    if (!err) *res = adminq->cq[cid].cs;
    else ERROR();
    return err;
}

/**
 * NVMe set features command.
 * Submit the command and wait for completion.
 * @param   dev         device context
 * @param   nsid        namespace id
 * @param   fid         feature id
 * @param   prp1        PRP1 address
 * @param   prp2        PRP2 address
 * @param   res         dword 0 value returned
 * @return  completion status (0 if ok).
 */
int nvme_acmd_set_features(nvme_device_t* dev, int nsid,
                           int fid, u64 prp1, u64 prp2, u32* res)
{
    nvme_queue_t* adminq = &dev->adminq;
    int cid = adminq->sq_tail;
    nvme_acmd_set_features_t* cmd = &adminq->sq[cid].set_features;

    memset(cmd, 0, sizeof (*cmd));
    cmd->common.opc = NVME_ACMD_SET_FEATURES;
    cmd->common.cid = cid;
    cmd->common.nsid = nsid;
    cmd->common.prp1 = prp1;
    cmd->common.prp2 = prp2;
    cmd->fid = fid;
    cmd->val = *res;
    *res = -1;

    DEBUG_FN("cid=%#x fid=%d", cid, fid);
    int err = nvme_submit_cmd(adminq);
    if (!err) err = nvme_wait_completion(adminq, cid, 10);
    if (!err) *res = adminq->cq[cid].cs;
    else ERROR();
    return err;
}

/**
 * NVMe create I/O completion queue command.
 * Submit the command and wait for completion.
 * @param   ioq         io queue
 * @param   prp         PRP1 address
 * @return  0 if ok, else -1.
 */
int nvme_acmd_create_cq(nvme_queue_t* ioq, u64 prp)
{
    nvme_queue_t* adminq = &ioq->dev->adminq;
    int cid = adminq->sq_tail;
    nvme_acmd_create_cq_t* cmd = &adminq->sq[cid].create_cq;

    memset(cmd, 0, sizeof (*cmd));
    cmd->common.opc = NVME_ACMD_CREATE_CQ;
    cmd->common.cid = cid;
    cmd->common.prp1 = prp;
    cmd->pc = 1;
    cmd->qid = ioq->id;
    cmd->qsize = ioq->size - 1;

    DEBUG_FN("q=%d cid=%#x qs=%d", ioq->id, cid, ioq->size);
    int err = nvme_submit_cmd(adminq);
    if (!err) err = nvme_wait_completion(adminq, cid, 10);
    if (err) ERROR();
    return err;
}

/**
 * NVMe create I/O submission queue command.
 * Submit the command and wait for completion.
 * @param   ioq         io queue
 * @param   prp         PRP1 address
 * @return  0 if ok, else -1.
 */
int nvme_acmd_create_sq(nvme_queue_t* ioq, u64 prp)
{
    nvme_queue_t* adminq = &ioq->dev->adminq;
    int cid = adminq->sq_tail;
    nvme_acmd_create_sq_t* cmd = &adminq->sq[cid].create_sq;

    memset(cmd, 0, sizeof (*cmd));
    cmd->common.opc = NVME_ACMD_CREATE_SQ;
    cmd->common.cid = cid;
    cmd->common.prp1 = prp;
    cmd->pc = 1;
    cmd->qprio = 2; // 0=urgent 1=high 2=medium 3=low
    cmd->qid = ioq->id;
    cmd->cqid = ioq->id;
    cmd->qsize = ioq->size - 1;

    DEBUG_FN("q=%d cid=%#x qs=%d", ioq->id, cid, ioq->size);
    int err = nvme_submit_cmd(adminq);
    if (!err) err = nvme_wait_completion(adminq, cid, 10);
    if (err) ERROR();
    return err;
}

/**
 * NVMe delete I/O queue command.
 * Submit the command and wait for completion.
 * @param   ioq         io queue
 * @param   opc         op code
 * @return  0 if ok else error code.
 */
static inline int nvme_acmd_delete_ioq(nvme_queue_t* ioq, int opc)
{
    nvme_queue_t* adminq = &ioq->dev->adminq;
    int cid = adminq->sq_tail;
    nvme_acmd_delete_ioq_t* cmd = &adminq->sq[cid].delete_ioq;

    memset(cmd, 0, sizeof (*cmd));
    cmd->common.opc = opc;
    cmd->common.cid = cid;
    cmd->qid = ioq->id;

    DEBUG_FN("%cq=%d cid=%#x",
             opc == NVME_ACMD_DELETE_CQ ? 'c' : 's', ioq->id, cid);
    int err = nvme_submit_cmd(adminq);
    if (!err) err = nvme_wait_completion(adminq, cid, 10);
    if (err) ERROR();
    return err;
}

/**
 * NVMe delete I/O submission queue command.
 * Submit the command and wait for completion.
 * @param   ioq         io queue
 * @return  0 if ok else error code.
 */
int nvme_acmd_delete_sq(nvme_queue_t* ioq)
{
    return nvme_acmd_delete_ioq(ioq, NVME_ACMD_DELETE_SQ);
}

/**
 * NVMe delete I/O completion queue command.
 * Submit the command and wait for completion.
 * @param   ioq         io queue
 * @return  0 if ok else error code.
 */
int nvme_acmd_delete_cq(nvme_queue_t* ioq)
{
    return nvme_acmd_delete_ioq(ioq, NVME_ACMD_DELETE_CQ);
}

/**
 * NVMe submit a vendor specific command.
 * @param   dev         device context
 * @param   nsid        namespace
 * @param   prp1        PRP1 address
 * @param   prp2        PRP2 address
 * @param   opc         vendor specific op code
 * @param   cdw10_15    command dwords 10-15
 * @return  0 if ok else -1.
 */
int nvme_acmd_vs(nvme_device_t* dev, int nsid, u64 prp1, u64 prp2,
                 int opc, u32 cdw10_15[6])
{
    nvme_queue_t* adminq = &dev->adminq;
    return nvme_cmd_vs(adminq, adminq->sq_tail, nsid, prp1, prp2, opc, cdw10_15);
}

/**
 * NVMe submit a vendor specific command.
 * @param   ioq         io queue
 * @param   cid         command id
 * @param   nsid        namespace
 * @param   prp1        PRP1 address
 * @param   prp2        PRP2 address
 * @param   opc         vendor specific op code
 * @param   cdw10_15    command dwords 10-15
 * @return  0 if ok else -1.
 */
int nvme_cmd_vs(nvme_queue_t* ioq, u16 cid, int nsid, u64 prp1, u64 prp2,
                int opc, u32 cdw10_15[6])
{
    nvme_command_vs_t* cmd = &ioq->sq[ioq->sq_tail].vs;

    memset(cmd, 0, sizeof (*cmd));
    cmd->common.opc = opc;
    cmd->common.cid = cid;
    cmd->common.nsid = nsid;
    cmd->common.prp1 = prp1;
    cmd->common.prp2 = prp2;
    if (cdw10_15) memcpy(cmd->cdw10_15, cdw10_15, sizeof(cmd->cdw10_15));
    DEBUG_FN("q=%d t=%d h=%d cid=%#x nsid=%d opc=%#x",
             ioq->id, ioq->sq_tail, ioq->sq_head, cid, nsid, opc);
    return nvme_submit_cmd(ioq);
}

/**
 * NVMe submit a read write command.
 * @param   ioq         io queue
 * @param   opc         op code
 * @param   cid         command id
 * @param   nsid        namespace
 * @param   slba        startling logical block address
 * @param   nlb         number of logical blocks
 * @param   prp1        PRP1 address
 * @param   prp2        PRP2 address
 * @return  0 if ok else -1.
 */
int nvme_cmd_rw(nvme_queue_t* ioq, int opc, u16 cid, int nsid,
                u64 slba, int nlb, u64 prp1, u64 prp2)
{
    nvme_command_rw_t* cmd = &ioq->sq[ioq->sq_tail].rw;

    memset(cmd, 0, sizeof (*cmd));
    cmd->common.opc = opc;
    cmd->common.cid = cid;
    cmd->common.nsid = nsid;
    cmd->common.prp1 = prp1;
    cmd->common.prp2 = prp2;
    cmd->slba = slba;
    cmd->nlb = nlb - 1;
    DEBUG_FN("q=%d t=%d h=%d cid=%#x nsid=%d lba=%#lx nb=%#x (%c)",
             ioq->id, ioq->sq_tail, ioq->sq_head, cid, nsid, slba, nlb,
             opc == NVME_CMD_READ? 'R' : 'W');
    return nvme_submit_cmd(ioq);
}

/**
 * NVMe submit a read command.
 * @param   ioq         io queue
 * @param   cid         command id
 * @param   nsid        namespace
 * @param   slba        startling logical block address
 * @param   nlb         number of logical blocks
 * @param   prp1        PRP1 address
 * @param   prp2        PRP2 address
 * @return  0 if ok else -1.
 */
int nvme_cmd_read(nvme_queue_t* ioq, u16 cid, int nsid,
                  u64 slba, int nlb, u64 prp1, u64 prp2)
{
    return nvme_cmd_rw(ioq, NVME_CMD_READ, cid, nsid, slba, nlb, prp1, prp2);
}

/**
 * NVMe submit a write command.
 * @param   ioq         io queue
 * @param   cid         command id
 * @param   nsid        namespace
 * @param   slba        startling logical block address
 * @param   nlb         number of blocks
 * @param   prp1        PRP1 address
 * @param   prp2        PRP2 address
 * @return  0 if ok else -1.
 */
int nvme_cmd_write(nvme_queue_t* ioq, u16 cid, int nsid,
                   u64 slba, int nlb, u64 prp1, u64 prp2)
{
    return nvme_cmd_rw(ioq, NVME_CMD_WRITE, cid, nsid, slba, nlb, prp1, prp2);
}

/**
 * Create an IO submission-completion queue pair.
 * @param   dev         device context
 * @param   ioq         if NULL then allocate queue
 * @param   id          queue id
 * @param   qsize       queue size
 * @param   sqbuf       submission queue buffer
 * @param   sqpa        submission queue IO physical address
 * @param   cqbuf       completion queue buffer
 * @param   cqpa        admin completion IO physical address
 * @return  pointer to the created io queue or NULL if failure.
 */
nvme_queue_t* nvme_create_ioq(nvme_device_t* dev, nvme_queue_t* ioq,
            int id, int qsize, void* sqbuf, u64 sqpa, void* cqbuf, u64 cqpa)
{
    if (!ioq) ioq = zalloc(sizeof(*ioq));
    else ioq->ext = 1;

    ioq->dev = dev;
    ioq->id = id;
    ioq->size = qsize;
    ioq->sq = sqbuf;
    ioq->cq = cqbuf;
    ioq->sq_doorbell = dev->reg->sq0tdbl + (2 * id * dev->dbstride);
    ioq->cq_doorbell = ioq->sq_doorbell + dev->dbstride;

    if (nvme_acmd_create_cq(ioq, cqpa) || nvme_acmd_create_sq(ioq, sqpa)) {
        free(ioq);
        return NULL;
    }
    return ioq;
}

/**
 * Delete an IO submission-completion queue pair.
 * @param   ioq         io queue to setup
 * @return  0 if ok else -1.
 */
int nvme_delete_ioq(nvme_queue_t* ioq)
{
    if (!ioq) return -1;
    if (nvme_acmd_delete_sq(ioq) || nvme_acmd_delete_cq(ioq)) return -1;
    if (!ioq->ext) free(ioq);
    return 0;
}

/**
 * NVMe setup admin submission-completion queue pair.
 * @param   dev         device context
 * @param   qsize       queue size
 * @param   sqbuf       submission queue buffer
 * @param   sqpa        submission queue IO physical address
 * @param   cqbuf       completion queue buffer
 * @param   cqpa        admin completion physical address
 * @return  pointer to the admin queue or NULL if failure.
 */
nvme_queue_t* nvme_setup_adminq(nvme_device_t* dev, int qsize,
                                void* sqbuf, u64 sqpa, void* cqbuf, u64 cqpa)
{
    if (nvme_ctlr_disable(dev)) return NULL;

    nvme_queue_t* adminq = &dev->adminq;
    adminq->dev = dev;
    adminq->id = 0;
    adminq->size = qsize;
    adminq->sq = sqbuf;
    adminq->cq = cqbuf;
    adminq->sq_doorbell = dev->reg->sq0tdbl;
    adminq->cq_doorbell = adminq->sq_doorbell + dev->dbstride;

    nvme_adminq_attr_t aqa;
    aqa.val = 0;
    aqa.asqs = aqa.acqs = qsize - 1;
    w32(dev, &dev->reg->aqa.val, aqa.val);
    w64(dev, &dev->reg->asq, sqpa);
    w64(dev, &dev->reg->acq, cqpa);

    nvme_controller_config_t cc;
    cc.val = 0;
    cc.shn = 0;
    cc.ams = 0;
    cc.css = 0;
    cc.iosqes = 6;
    cc.iocqes = 4;
    cc.mps = dev->pageshift - 12;
    if (nvme_ctlr_enable(dev, cc)) return NULL;

    DEBUG_FN("qsize=%d cc=%#x aqa=%#x asq=%#lx acq=%#lx",
             qsize, cc.val, aqa.val, sqpa, cqpa);
    DEBUG_FN("vs=%#x intms=%#x intmc=%#x csts=%#x", dev->reg->vs.val,
             dev->reg->intms, dev->reg->intmc, dev->reg->csts.val);
    return adminq;
}

/**
 * Create an NVMe device context and map the controller register.
 * @param   dev         if NULL then allocate context
 * @param   mapfd       file descriptor for register mapping
 * @return  device context or NULL if failure.
 */
nvme_device_t* nvme_create(nvme_device_t* dev, int mapfd)
{
    if (!dev) dev = zalloc(sizeof(*dev));
    else dev->ext = 1;

    dev->reg = mmap(0, sizeof(nvme_controller_reg_t),
                    PROT_READ|PROT_WRITE, MAP_SHARED|MAP_LOCKED, mapfd, 0);
    if (dev->reg == MAP_FAILED) {
        ERROR("mmap: %s", strerror(errno));
        return NULL;
    }

    nvme_controller_cap_t cap;
    cap.val = r64(dev, &dev->reg->cap.val);
    dev->timeout = cap.to;              // in 500ms units
    dev->mpsmin = cap.mpsmin;           // 2 ^ (12 + MPSMIN)
    dev->mpsmax = cap.mpsmax;           // 2 ^ (12 + MPSMAX)
    dev->pageshift = 12 + dev->mpsmin;
    dev->maxqsize = cap.mqes + 1;
    dev->dbstride = 1 << cap.dstrd;     // in u32 size offset

    nvme_rdtsec = rdtsc_second();

    DEBUG_FN("cap=%#lx ps=%u-%u to=%u maxqs=%u dbs=%u", cap.val,
             cap.mpsmin, cap.mpsmax, cap.to, dev->maxqsize, dev->dbstride);

    return dev;
}

/**
 * Delete an NVMe device context
 * @param   dev         device context
 */
void nvme_delete(nvme_device_t* dev)
{
    if (dev && dev->reg) {
        if (munmap(dev->reg, sizeof(nvme_controller_reg_t))) {
            ERROR("munmap: %s", strerror(errno));
        }
    }
    if (!dev->ext) free(dev);
}

