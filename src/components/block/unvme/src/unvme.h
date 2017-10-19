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
 * @brief UNVMe client header file.
 */

#ifndef _UNVME_H
#define _UNVME_H

#include <stdint.h>

#ifndef _U_TYPE
#define _U_TYPE                     ///< bit size data types
typedef int8_t          s8;         ///< 8-bit signed
typedef int16_t         s16;        ///< 16-bit signed
typedef int32_t         s32;        ///< 32-bit signed
typedef int64_t         s64;        ///< 64-bit signed
typedef uint8_t         u8;         ///< 8-bit unsigned
typedef uint16_t        u16;        ///< 16-bit unsigned
typedef uint32_t        u32;        ///< 32-bit unsigned
typedef uint64_t        u64;        ///< 64-bit unsigned
#endif // _U_TYPE

#define UNVME_TIMEOUT   30          ///< default I/O timeout in seconds
#define UNVME_QSIZE     256         ///< default I/O queue size

/// Namespace attributes structure
typedef struct _unvme_ns {
    u32                 pci;        ///< PCI device id
    u16                 id;         ///< namespace id
    u16                 vid;        ///< vendor id
    char                device[16]; ///< PCI device name (BB:DD.F/N)
    char                mn[40];     ///< model number
    char                sn[20];     ///< serial number
    char                fr[8];      ///< firmware revision
    u64                 blockcount; ///< total number of available blocks
    u64                 pagecount;  ///< total number of available pages
    u16                 blocksize;  ///< logical block size
    u16                 pagesize;   ///< page size
    u16                 blockshift; ///< block size shift value
    u16                 pageshift;  ///< page size shift value
    u16                 bpshift;    ///< block to page shift
    u16                 nbpp;       ///< number of blocks per page
    u16                 maxbpio;    ///< max number of blocks per I/O
    u16                 maxppio;    ///< max number of pages per I/O
    u16                 maxiopq;    ///< max number of I/O submissions per queue
    u16                 qcount;     ///< number of I/O queues
    u16                 maxqcount;  ///< max number of queues supported
    u16                 qsize;      ///< I/O queue size
    u16                 maxqsize;   ///< max queue size supported
    u16                 nscount;    ///< number of namespaces available
    void*               ses;        ///< associated session
} unvme_ns_t;

/// I/O descriptor (not to be copied and cleared upon apoll completion)
typedef struct _unvme_iod {
    void*               buf;        ///< data buffer (submitted)
    u64                 slba;       ///< starting lba (submitted)
    u32                 nlb;        ///< number of blocks (submitted)
    u32                 qid;        ///< queue id (submitted)
    u32                 opc;        ///< op code
    u32                 id;         ///< descriptor id
} *unvme_iod_t;

// Export functions
const unvme_ns_t* unvme_open(const char* pciname);
const unvme_ns_t* unvme_openq(const char* pciname, int qcount, int qsize);
int unvme_close(const unvme_ns_t* ns);

void* unvme_alloc(const unvme_ns_t* ns, u64 size);
int unvme_free(const unvme_ns_t* ns, void* buf);

int unvme_write(const unvme_ns_t* ns, int qid, const void* buf, u64 slba, u32 nlb);
int unvme_read(const unvme_ns_t* ns, int qid, void* buf, u64 slba, u32 nlb);

unvme_iod_t unvme_awrite(const unvme_ns_t* ns, int qid, const void* buf, u64 slba, u32 nlb);
unvme_iod_t unvme_aread(const unvme_ns_t* ns, int qid, void* buf, u64 slba, u32 nlb);
int unvme_apoll(unvme_iod_t iod, int timeout);
int unvme_apoll_cs(unvme_iod_t iod, int timeout, u32* cqe_cs);

#endif // _UNVME_H

