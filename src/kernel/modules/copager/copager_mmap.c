/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Daniel Waddington (daniel.waddington@ibm.com)
 * Feng Li (feng.li1@ibm.com)
 *
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/semaphore.h>
#include <linux/init.h>
#include <linux/poll.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/idr.h>
#include <linux/kobject.h>
#include <linux/cdev.h>
#include <linux/miscdevice.h>
#include <linux/vmalloc.h>
#include <linux/mman.h>
#include <linux/pfn_t.h>
#include <linux/kfifo.h>
#include <linux/mutex.h>
#include <linux/pagemap.h>
#include <asm/current.h>
#include <asm/tlbflush.h>

#include "copager_msg.h"

typedef unsigned long addr_t;

extern DECLARE_KFIFO(srq, service_request_t*, FIFO_SIZE);
extern wait_queue_head_t wq;

/*
 * copager_vma_open
 *
 * Called when a device mapping is created via mmap
 * ,fork, munmap, etc.).  Increments the reference count on the
 * underlying mspec data so it is not freed prematurely.
 *
 * to do: support multiple threads by refcount, see driver/msepc
 */
void copager_vma_open(struct vm_area_struct *vma)
{
  //  struct vma_data *vdata;
}

// reclaim the pysical pages
void copager_vma_close(struct vm_area_struct *vma)
{
}

/*
 * copager fault handler
 *
 * @ vma: current mapped vma
 * @ vmf: fault info, filled by kernel mm
 * to support multithreaded
 */
int copager_vma_fault(struct vm_fault * vmf){

  //  addr_t virt_addr_faulted = (addr_t) vmf->virtual_address;
  addr_t virt_addr_faulted = vmf->address;

#ifdef DEBUG
  printk(KERN_NOTICE "[copager]: PAGE-FAULT flags %x, vma flag %lx, usr_virt add %016lx\n",
         vmf->flags, vmf->vma->vm_flags, (unsigned long)vmf->address);
#endif
 
  {
    unsigned long timeout;
    volatile int signal = 0;
    service_request_t sr;
    sr.addr[0] = virt_addr_faulted;
    sr.pid = current->pid;
    sr.signal = &signal;
    sr.this_ptr = &sr;

    /* post request */
    kfifo_put(&srq, &sr);

    /* wake threads */
    wake_up(&wq);

    /* wait for response */
    timeout = jiffies + (HZ*5);// 10s timout
    while(!signal) {
      if(jiffies > timeout) {
        break;
      }
    }
    
    if(!signal) {
      printk(KERN_INFO "[copager]: PF handler timed out");
      return VM_FAULT_RETRY; /* timed out */
    }

#ifdef DEBUG
    printk(KERN_INFO "[copager]: PF thread signalled from userspace\n");
    printk(KERN_INFO "[copager]: addr[PHYS]=%lx addr[INVAL]=%lx pid=%d\n",
           sr.addr[ADDR_IDX_PHYS], sr.addr[ADDR_IDX_INVAL], sr.pid);
#endif
    
    BUG_ON(sr.pid != current->pid);
    if(sr.addr[ADDR_IDX_PHYS] == 0)
       return VM_FAULT_SIGBUS;

    /* check results */
    if(sr.addr[ADDR_IDX_INVAL] & 0xfff ||
       sr.addr[ADDR_IDX_PHYS] & 0xfff) { /* todo extend to 2M pages */
      printk(KERN_NOTICE "invalid address; not page aligned");
      return VM_FAULT_SIGBUS;
    }

#ifdef DEBUG
    printk(KERN_INFO "[copager]: updating page table (%lx->%lx)\n",
           virt_addr_faulted, sr.addr[ADDR_IDX_PHYS]);            
#endif
    
    /* update page table */
    {
      pfn_t pfn = phys_to_pfn_t(sr.addr[ADDR_IDX_PHYS], PFN_DEV | PFN_MAP);
      vm_insert_pfn(vmf->vma, virt_addr_faulted, pfn_t_to_pfn(pfn));
    }

    /* optionally evict page - currently max 1 evication */
    if(sr.addr[ADDR_IDX_INVAL]) {      
      int ret = zap_vma_ptes(vmf->vma, sr.addr[ADDR_IDX_INVAL], PAGE_SIZE); /* zap will invalidate TLB */
      if(ret){
        printk(KERN_NOTICE "zap not sucessful, return %d\n", ret);
      }
    }
    
    return VM_FAULT_NOPAGE;
    //return VM_FAULT_SIGBUS; // testing only
           
  }

  return VM_FAULT_RETRY; /* timed out */
} 

static struct vm_operations_struct copager_remap_vm_ops={
  .open = copager_vma_open,
  .close = copager_vma_close,
  .fault = copager_vma_fault,
};



// map device memory to userspace address 
int copager_fops_mmap(struct file* file, struct vm_area_struct *vma)
{
  //  struct vma_data * vdata;

  // requirement for vm_insert_page
  //vma->vm_flags |= VM_MIXEDMAP|VM_PFNMAP;
  //vma->vm_flags |= VM_MIXEDMAP|VM_PFNMAP;
  //vma->vm_flags |= VM_MIXEDMAP;
  vma->vm_flags |= VM_PFNMAP;
  vma->vm_flags &= ~(VM_MAYWRITE|VM_MAYREAD|VM_MAYEXEC);

  // set backend of vma
  vma->vm_ops = &copager_remap_vm_ops;

  //  vdata = kmalloc(sizeof(struct vma_data), GFP_KERNEL);
  //  vma->vm_private_data = vdata;

  copager_vma_open(vma);

#ifdef DEBUG
  printk(KERN_INFO "copager: mmap virt=%016lx, length %lx pgoff=%lx, vm_flags = %lx",
         vma->vm_start,
         vma->vm_end- vma->vm_start,
         vma->vm_pgoff,
         vma->vm_flags);
#endif
  
  return 0;
}


