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
 * @brief VFIO function support header files.
 */

#ifndef _UNVME_VFIO_H
#define _UNVME_VFIO_H

#include <stdlib.h>
#include <pthread.h>
#include <linux/vfio.h>

/// VFIO dma allocation structure
typedef struct _vfio_dma {
    void*                   buf;        ///< memory buffer
    size_t                  size;       ///< allocated size
    __u64                   addr;       ///< I/O DMA address
    struct _vfio_mem*       mem;        ///< private mem
} vfio_dma_t;

/// VFIO memory allocation entry
typedef struct _vfio_mem {
    struct _vfio_device*    dev;        ///< device owner
    int                     mmap;       ///< mmap indication flag
    vfio_dma_t              dma;        ///< dma mapped memory
    size_t                  size;       ///< size
    struct _vfio_mem*       prev;       ///< previous entry
    struct _vfio_mem*       next;       ///< next entry
} vfio_mem_t;

/// VFIO device structure
typedef struct _vfio_device {
    int                     pci;        ///< PCI device number
    int                     fd;         ///< device descriptor
    int                     groupfd;    ///< group file descriptor
    int                     contfd;     ///< container file descriptor
    int                     msixsize;   ///< max MSIX table size
    int                     msixnvec;   ///< number of enabled MSIX vectors
    int                     ext;        ///< externally allocated flag
    __u64                   iovabase;   ///< next DMA (virtual IO) address
    __u64                   iovanext;   ///< next DMA (virtual IO) address
    pthread_mutex_t         lock;       ///< multithreaded lock
    vfio_mem_t*             memlist;    ///< memory allocated list
} vfio_device_t;

// Export functions
vfio_device_t* vfio_create(vfio_device_t* dev, int pci);
void vfio_delete(vfio_device_t* dev);
void vfio_msix_enable(vfio_device_t* dev, int start, int nvec, __s32* efds);
void vfio_msix_disable(vfio_device_t* dev);
int vfio_mem_free(vfio_mem_t* mem);
vfio_dma_t* vfio_dma_map(vfio_device_t* dev, size_t size, void* pmb);
int vfio_dma_unmap(vfio_dma_t* dma);
vfio_dma_t* vfio_dma_alloc(vfio_device_t* dev, size_t size);
int vfio_dma_free(vfio_dma_t* dma);

#endif // _UNVME_VFIO_H

