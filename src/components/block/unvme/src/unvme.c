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
 * @brief UNVMe client library interface functions.
 */

#include "unvme_core.h"

/**
 * Open a client session with specified number of IO queues and queue size.
 * @param   pciname     PCI device name (as %x:%x.%x[/NSID] format)
 * @param   qcount      number of io queues
 * @param   qsize       io queue size
 * @return  namespace pointer or NULL if error.
 */
const unvme_ns_t* unvme_openq(const char* pciname, int qcount, int qsize)
{
    if (qcount < 0 || qsize < 0 || qsize == 1) {
        ERROR("invalid qcount %d or qsize %d", qcount, qsize);
        return NULL;
    }

    int b, d, f, nsid = 1;
    if ((sscanf(pciname, "%x:%x.%x/%x", &b, &d, &f, &nsid) != 4) &&
        (sscanf(pciname, "%x:%x.%x", &b, &d, &f) != 3)) {
        ERROR("invalid PCI %s (expect %%x:%%x.%%x[/NSID] format)", pciname);
        return NULL;
    }
    int pci = (b << 16) + (d << 8) + f;

    return unvme_do_open(pci, nsid, qcount, qsize);
}

/**
 * Open a client session.
 * @param   pciname     PCI device name (as %x:%x.%x[/NSID] format)
 * @return  namespace pointer or NULL if error.
 */
const unvme_ns_t* unvme_open(const char* pciname)
{
    return unvme_openq(pciname, 0, 0);
}

/**
 * Close a client session and delete its contained io queues.
 * @param   ns          namespace handle
 * @return  0 if ok else error code.
 */
int unvme_close(const unvme_ns_t* ns)
{
    return unvme_do_close(ns);
}

/**
 * Allocate an I/O buffer associated with a session.
 * @param   ns          namespace handle
 * @param   size        buffer size
 * @return  the allocated buffer or NULL if failure.
 */
void* unvme_alloc(const unvme_ns_t* ns, u64 size)
{
    return unvme_do_alloc(ns, size);
}

/**
 * Free an I/O buffer associated with a session.
 * @param   ns          namespace handle
 * @param   buf         buffer pointer
 * @return  0 if ok else -1.
 */
int unvme_free(const unvme_ns_t* ns, void* buf)
{
    return unvme_do_free(ns, buf);
}

/**
 * Read data from specified logical blocks on device.
 * @param   ns          namespace handle
 * @param   qid         client queue index
 * @param   buf         data buffer (from unvme_alloc)
 * @param   slba        starting logical block
 * @param   nlb         number of logical blocks
 * @return  I/O descriptor or NULL if failed.
 */
unvme_iod_t unvme_aread(const unvme_ns_t* ns, int qid, void* buf, u64 slba, u32 nlb)
{
    return (unvme_iod_t)unvme_rw(ns, qid, NVME_CMD_READ, buf, slba, nlb);
}

/**
 * Write data to specified logical blocks on device.
 * @param   ns          namespace handle
 * @param   qid         client queue index
 * @param   buf         data buffer (from unvme_alloc)
 * @param   slba        starting logical block
 * @param   nlb         number of logical blocks
 * @return  I/O descriptor or NULL if failed.
 */
unvme_iod_t unvme_awrite(const unvme_ns_t* ns, int qid,
                         const void* buf, u64 slba, u32 nlb)
{
    return (unvme_iod_t)unvme_rw(ns, qid, NVME_CMD_WRITE, (void*)buf, slba, nlb);
}

/**
 * Poll for completion status of a previous IO submission.
 * If there's no error, the descriptor will be freed.
 * @param   iod         IO descriptor
 * @param   timeout     in seconds
 * @return  0 if ok else error status (-1 for timeout).
 */
int unvme_apoll(unvme_iod_t iod, int timeout)
{
    return unvme_do_poll((unvme_desc_t*)iod, timeout, NULL);
}

/**
 * Poll for completion status of a previous IO submission.
 * If there's no error, the descriptor will be freed.
 * @param   iod         IO descriptor
 * @param   timeout     in seconds
 * @param   cqe_cs      CQE command specific DW0 returned
 * @return  0 if ok else error status (-1 for timeout).
 */
int unvme_apoll_cs(unvme_iod_t iod, int timeout, u32* cqe_cs)
{
    return unvme_do_poll((unvme_desc_t*)iod, timeout, cqe_cs);
}

/**
 * Read data from specified logical blocks on device.
 * @param   ns          namespace handle
 * @param   qid         client queue index
 * @param   buf         data buffer (from unvme_alloc)
 * @param   slba        starting logical block
 * @param   nlb         number of logical blocks
 * @return  0 if ok else error status.
 */
int unvme_read(const unvme_ns_t* ns, int qid, void* buf, u64 slba, u32 nlb)
{
    unvme_desc_t* desc = unvme_rw(ns, qid, NVME_CMD_READ, buf, slba, nlb);
    if (desc) {
        sched_yield();
        return unvme_do_poll(desc, UNVME_TIMEOUT, NULL);
    }
    return -1;
}

/**
 * Write data to specified logical blocks on device.
 * @param   ns          namespace handle
 * @param   qid         client queue index
 * @param   buf         data buffer (from unvme_alloc)
 * @param   slba        starting logical block
 * @param   nlb         number of logical blocks
 * @return  0 if ok else error status.
 */
int unvme_write(const unvme_ns_t* ns, int qid,
                const void* buf, u64 slba, u32 nlb)
{
    unvme_desc_t* desc = unvme_rw(ns, qid, NVME_CMD_WRITE, (void*)buf, slba, nlb);
    if (desc) {
        sched_yield();
        return unvme_do_poll(desc, UNVME_TIMEOUT, NULL);
    }
    return -1;
}

